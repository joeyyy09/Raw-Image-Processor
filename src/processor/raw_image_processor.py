import rawpy
import numpy as np
import cv2
from pathlib import Path
import time
import json
import keyboard
import logging
from datetime import datetime
from .memory_manager import MemoryManager
from .display_manager import DisplayManager
from .settings_manager import SettingsManager
from .image_enhancements import apply_adaptive_noise_reduction, apply_smart_sharpening

class RAWImageProcessor:
    def __init__(self):
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.settings_file = 'processing_settings.json'
        self.settings_manager = SettingsManager(self.settings_file)
        self.settings = self.settings_manager.load_settings()
        self.processing_cancelled = False
        self.is_processing = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Memory thresholds (in percentage)
        self.memory_manager = MemoryManager(self.logger)
        
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
                        processed_bgr = apply_adaptive_noise_reduction(processed_bgr)
                    
                    if self.settings['sharpness'] > 0:
                        self.logger.info("Applying content-aware sharpening...")
                        processed_bgr = apply_smart_sharpening(processed_bgr)
                    
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
                display_manager = DisplayManager(self.logger)
                display_manager.display_images(preview_bgr, processed_bgr)
            
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

    def get_cache_path(self, raw_path):
        """Generate cache path for processed image"""
        return self.cache_dir / f"{Path(raw_path).stem}_processed.npz"

    def cancel_processing(self):
        """Cancel the current processing operation"""
        if self.is_processing:
            self.logger.info("Cancelling processing...")
            self.processing_cancelled = True

    def check_image_size(self, image):
        """Check if image needs downsampling based on size and memory"""
        total_pixels = image.shape[0] * image.shape[1]
        if (total_pixels > self.settings['downsample_threshold'] or 
            self.memory_manager.monitor_memory() > self.memory_manager.memory_threshold):
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