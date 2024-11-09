import rawpy
import numpy as np
import cv2
from pathlib import Path
import sys
import time
import json
import os
import psutil
from datetime import datetime
import threading
import keyboard
import logging

class RAWImageProcessor:
    def __init__(self):
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.settings_file = 'processing_settings.json'
        self.load_settings()
        self.processing_cancelled = False
        self.is_processing = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Memory thresholds (in percentage)
        self.memory_threshold = 80
        self.memory_critical = 90
        
        # Power mode
        self.power_saving_mode = self.check_battery_status()
        
    def check_battery_status(self):
        """Check if running on battery and its status"""
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                return battery.power_plugged is False and battery.percent < 20
            return False
        except:
            return False

    def load_settings(self):
        """Load processing settings from JSON or use defaults"""
        try:
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
        except:
            self.settings = {
                'brightness': 1.2,
                'contrast': 1.1,
                'sharpness': 1.0,
                'noise_reduction': True,
                'auto_wb': True,
                'preview_size': (800, 600),
                'save_metadata': True,
                'downsample_threshold': 50000000,  # Pixel count threshold for downsampling
                'power_saving_enabled': True
            }
            self.save_settings()

    def monitor_memory(self):
        """Monitor system memory usage"""
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_critical:
            self.logger.warning("Critical memory usage detected! Attempting cleanup...")
            self.cleanup_memory()
        return memory.percent

    def cleanup_memory(self):
        """Perform memory cleanup operations"""
        import gc
        gc.collect()
        
        # Clear opencv window caches
        cv2.destroyAllWindows()
        
        # Clear any unnecessary cached files
        if self.cache_dir.exists():
            files = list(self.cache_dir.glob('*'))
            if len(files) > 10:  # Keep only 10 most recent files
                files.sort(key=lambda x: x.stat().st_mtime)
                for f in files[:-10]:
                    f.unlink()

    def check_image_size(self, image):
        """Check if image needs downsampling based on size and memory"""
        total_pixels = image.shape[0] * image.shape[1]
        if (total_pixels > self.settings['downsample_threshold'] or 
            self.monitor_memory() > self.memory_threshold):
            scale_factor = np.sqrt(self.settings['downsample_threshold'] / total_pixels)
            new_size = (int(image.shape[1] * scale_factor), 
                       int(image.shape[0] * scale_factor))
            return cv2.resize(image, new_size)
        return image

    def setup_interactive_controls(self):
        """Setup interactive parameter adjustment"""
        def on_brightness(value):
            self.settings['brightness'] = value / 100.0
        
        def on_contrast(value):
            self.settings['contrast'] = value / 100.0
        
        def on_sharpness(value):
            self.settings['sharpness'] = value / 100.0

        cv2.namedWindow('Controls')
        cv2.createTrackbar('Brightness', 'Controls', 
                          int(self.settings['brightness'] * 100), 200, on_brightness)
        cv2.createTrackbar('Contrast', 'Controls', 
                          int(self.settings['contrast'] * 100), 200, on_contrast)
        cv2.createTrackbar('Sharpness', 'Controls', 
                          int(self.settings['sharpness'] * 100), 200, on_sharpness)

    def process_raw_image(self, raw_path):
        """Process a RAW image with optimizations and caching"""
        self.is_processing = True
        self.processing_cancelled = False
        start_time = time.time()
        cache_path = self.get_cache_path(raw_path)

        # Setup cancellation listener
        keyboard.on_press_key('esc', lambda _: self.cancel_processing())
        
        try:
            self.logger.info(f"Processing image: {raw_path}")
            
            # Update power saving mode
            self.power_saving_mode = (self.check_battery_status() and 
                                    self.settings['power_saving_enabled'])
            
            if self.power_saving_mode:
                self.logger.info("Running in power-saving mode")
                self.settings['noise_reduction'] = False
                self.settings['sharpness'] = 0.5
            
            # Check cache first
            if cache_path.exists():
                self.logger.info("Loading from cache...")
                with np.load(cache_path) as data:
                    processed_bgr = data['processed']
                    preview_bgr = data['preview']
            else:
                with rawpy.imread(raw_path) as raw:
                    if self.processing_cancelled:
                        raise InterruptedError("Processing cancelled by user")
                    
                    # Generate optimized preview
                    self.logger.info("Generating preview...")
                    preview = raw.postprocess(
                        half_size=True,
                        use_camera_wb=self.settings['auto_wb']
                    )
                    preview_resized = cv2.resize(preview, self.settings['preview_size'])
                    preview_bgr = cv2.cvtColor(preview_resized, cv2.COLOR_RGB2BGR)
                    
                    if self.processing_cancelled:
                        raise InterruptedError("Processing cancelled by user")
                    
                    # Full processing with optimizations
                    self.logger.info("Processing full image...")
                    processed = raw.postprocess(
                        use_camera_wb=self.settings['auto_wb'],
                        bright=self.settings['brightness'],
                        no_auto_bright=False,
                        gamma=(2.2, 4.5),
                        output_bps=8
                    )
                    
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    
                    # Check memory and downsample if needed
                    processed_bgr = self.check_image_size(processed_bgr)
                    
                    if self.settings['noise_reduction'] and not self.power_saving_mode:
                        self.logger.info("Applying adaptive noise reduction...")
                        processed_bgr = self.apply_adaptive_noise_reduction(processed_bgr)
                    
                    if self.settings['sharpness'] > 0:
                        self.logger.info("Applying content-aware sharpening...")
                        processed_bgr = self.apply_smart_sharpening(processed_bgr)
                    
                    if not self.processing_cancelled:
                        # Save to cache
                        self.logger.info("Saving to cache...")
                        np.savez_compressed(cache_path, 
                                         processed=processed_bgr,
                                         preview=preview_bgr)
            
            if not self.processing_cancelled:
                # Save the processed image with metadata
                output_path = 'processed_output.jpg'
                self.logger.info(f"Saving processed image to: {output_path}")
                cv2.imwrite(output_path, processed_bgr)
                
                if self.settings['save_metadata']:
                    self.save_metadata(raw_path, output_path, time.time() - start_time)
                
                # Setup interactive controls and display
                self.setup_interactive_controls()
                self.display_images(preview_bgr, processed_bgr)
            
        except InterruptedError as e:
            self.logger.info(str(e))
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            self.logger.error(f"Error type: {type(e)}")
            self.logger.error("Stack trace:", exc_info=True)
        finally:
            self.is_processing = False
            keyboard.unhook_all()
            cv2.destroyAllWindows()

    def cancel_processing(self):
        """Cancel the current processing operation"""
        if self.is_processing:
            self.logger.info("Cancelling processing...")
            self.processing_cancelled = True

    def display_images(self, preview_bgr, processed_bgr):
        """Display both preview and processed images"""
        try:
            # Get screen dimensions
            screen_width = 1920  # Default screen width
            screen_height = 1080  # Default screen height
            
            # Create windows
            cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
            
            # Calculate window sizes and positions
            preview_width = int(screen_width * 0.4)
            preview_height = int(preview_width * preview_bgr.shape[0] / preview_bgr.shape[1])
            
            processed_width = int(screen_width * 0.4)
            processed_height = int(processed_width * processed_bgr.shape[0] / processed_bgr.shape[1])
            
            # Position windows
            cv2.resizeWindow('Preview', preview_width, preview_height)
            cv2.moveWindow('Preview', int(screen_width * 0.05), 100)
            cv2.imshow('Preview', preview_bgr)
            
            cv2.resizeWindow('Processed Image', processed_width, processed_height)
            cv2.moveWindow('Processed Image', int(screen_width * 0.55), 100)
            cv2.moveWindow('Controls', int(screen_width * 0.4), 0)
            
            self.logger.info("\nPress 'ESC' to exit...")
            while cv2.getWindowProperty('Preview', cv2.WND_PROP_VISIBLE) > 0:
                cv2.imshow('Processed Image', processed_bgr)
                key = cv2.waitKey(100)
                if key == 27:  # ESC key
                    break
                
        except Exception as e:
            self.logger.error(f"Error displaying images: {str(e)}")
        finally:
            cv2.destroyAllWindows()
        
    def load_settings(self):
        """Load processing settings from JSON or use defaults"""
        try:
            with open(self.settings_file, 'r') as f:
                self.settings = json.load(f)
        except:
            self.settings = {
                'brightness': 1.2,
                'contrast': 1.1,
                'sharpness': 1.0,
                'noise_reduction': True,
                'auto_wb': True,
                'preview_size': (800, 600),
                'save_metadata': True
            }
            self.save_settings()
            
    def save_settings(self):
        """Save current settings to JSON"""
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)
            
    def get_cache_path(self, raw_path):
        """Generate cache path for processed image"""
        return self.cache_dir / f"{Path(raw_path).stem}_processed.npz"
            
    def process_raw_image(self, raw_path):
        """Process a RAW image with optimizations and caching"""
        start_time = time.time()
        cache_path = self.get_cache_path(raw_path)
        
        try:
            print(f"Processing image: {raw_path}")
            
            # Check cache first
            if cache_path.exists():
                print("Loading from cache...")
                with np.load(cache_path) as data:
                    processed_bgr = data['processed']
                    preview_bgr = data['preview']
            else:
                with rawpy.imread(raw_path) as raw:
                    # Generate optimized preview
                    print("Generating preview...")
                    preview = raw.postprocess(
                        half_size=True,
                        use_camera_wb=self.settings['auto_wb']
                    )
                    preview_resized = cv2.resize(preview, self.settings['preview_size'])
                    preview_bgr = cv2.cvtColor(preview_resized, cv2.COLOR_RGB2BGR)
                    
                    # Full processing with optimizations
                    print("Processing full image...")
                    processed = raw.postprocess(
                        use_camera_wb=self.settings['auto_wb'],
                        bright=self.settings['brightness'],
                        no_auto_bright=False,
                        gamma=(2.2, 4.5),
                        output_bps=8
                    )
                    
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
                    
                    # Apply enhancements based on settings
                    if self.settings['noise_reduction']:
                        print("Applying adaptive noise reduction...")
                        processed_bgr = self.apply_adaptive_noise_reduction(processed_bgr)
                    
                    if self.settings['sharpness'] > 0:
                        print("Applying content-aware sharpening...")
                        processed_bgr = self.apply_smart_sharpening(processed_bgr)
                    
                    # Save to cache
                    print("Saving to cache...")
                    np.savez_compressed(cache_path, 
                                     processed=processed_bgr,
                                     preview=preview_bgr)
            
            # Save the processed image with metadata
            output_path = 'processed_output.jpg'
            print(f"Saving processed image to: {output_path}")
            cv2.imwrite(output_path, processed_bgr)
            
            if self.settings['save_metadata']:
                self.save_metadata(raw_path, output_path, time.time() - start_time)
            
            # Display results in center of screen
            self.display_images(preview_bgr, processed_bgr)
            
            print("\nProcessing completed!")
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")
            print("\nPress any key to exit...")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print(f"Error type: {type(e)}")
            print("Stack trace:", sys.exc_info())
            
    def apply_adaptive_noise_reduction(self, image):
        """Apply noise reduction based on image characteristics"""
        # Calculate noise level
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_level = np.std(gray)
        
        # Adjust strength based on noise level
        h = int(noise_level * 2)  # Adaptive strength
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
        
    def apply_smart_sharpening(self, image):
        """Apply content-aware sharpening"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Adjust kernel based on image characteristics
        strength = min(1.5, max(0.5, 1.0 / (laplacian_var + 1e-5)))
        kernel = np.array([[-1,-1,-1], 
                         [-1, 9,-1],
                         [-1,-1,-1]]) * strength
        return cv2.filter2D(image, -1, kernel)
        
    def display_images(self, preview_bgr, processed_bgr):
        """Display both preview and processed images"""
        try:
            # Get screen dimensions
            screen_width = 1920  # Default screen width
            screen_height = 1080  # Default screen height
            
            # Create windows first
            cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
            
            # Calculate window sizes
            preview_width = int(screen_width * 0.4)
            preview_height = int(preview_width * preview_bgr.shape[0] / preview_bgr.shape[1])
            
            processed_width = int(screen_width * 0.4)
            processed_height = int(processed_width * processed_bgr.shape[0] / processed_bgr.shape[1])
            
            # Ensure windows aren't too tall
            if preview_height > screen_height * 0.8:
                preview_height = int(screen_height * 0.8)
                preview_width = int(preview_height * preview_bgr.shape[1] / preview_bgr.shape[0])
                
            if processed_height > screen_height * 0.8:
                processed_height = int(screen_height * 0.8)
                processed_width = int(processed_height * processed_bgr.shape[1] / processed_bgr.shape[0])
            
            # Position windows side by side
            preview_x = int(screen_width * 0.05)
            processed_x = int(screen_width * 0.55)
            y = (screen_height - max(preview_height, processed_height)) // 2
            
            # Set window positions and sizes
            cv2.resizeWindow('Preview', preview_width, preview_height)
            cv2.moveWindow('Preview', preview_x, y)
            cv2.imshow('Preview', preview_bgr)
            
            cv2.resizeWindow('Processed Image', processed_width, processed_height)
            cv2.moveWindow('Processed Image', processed_x, y)
            cv2.imshow('Processed Image', processed_bgr)
            
            # Wait for key press
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error displaying images: {str(e)}")
            # Continue processing even if display fails
        
    def save_metadata(self, input_path, output_path, processing_time):
        """Save processing metadata"""
        metadata = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'processing_time': processing_time,
            'processing_date': datetime.now().isoformat(),
            'settings_used': self.settings,
            'cache_used': self.get_cache_path(input_path).exists()
        }
        
        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

def main():
    processor = RAWImageProcessor()
    raw_path = "raw.ARW"
    
    if not Path(raw_path).exists():
        print(f"Error: Could not find {raw_path}")
        print("Make sure the RAW file is in the same directory as this script")
        return
    
    processor.process_raw_image(raw_path)
    sys.exit(0)

if __name__ == "__main__":
    main()


# from src.main import main

# if __name__ == "__main__":
#     main()