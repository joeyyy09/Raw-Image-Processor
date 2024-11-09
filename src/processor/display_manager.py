import cv2

class DisplayManager:
    def __init__(self, logger):
        self.logger = logger

    def display_images(self, preview_bgr, processed_bgr):
        """Display both preview and processed images"""
        try:
            screen_width = 1920
            screen_height = 1080
            
            cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Image', cv2.WINDOW_NORMAL)
            
            preview_width = int(screen_width * 0.4)
            preview_height = int(preview_width * preview_bgr.shape[0] / preview_bgr.shape[1])
            
            processed_width = int(screen_width * 0.4)
            processed_height = int(processed_width * processed_bgr.shape[0] / processed_bgr.shape[1])
            
            if preview_height > screen_height * 0.8:
                preview_height = int(screen_height * 0.8)
                preview_width = int(preview_height * preview_bgr.shape[1] / preview_bgr.shape[0])
            
            if processed_height > screen_height * 0.8:
                processed_height = int(screen_height * 0.8)
                processed_width = int(processed_height * processed_bgr.shape[1] / processed_bgr.shape[0])
            
            preview_x = int(screen_width * 0.05)
            processed_x = int(screen_width * 0.55)
            y = (screen_height - max(preview_height, processed_height)) // 2
            
            cv2.resizeWindow('Preview', preview_width, preview_height)
            cv2.moveWindow('Preview', preview_x, y)
            cv2.imshow('Preview', preview_bgr)
            
            cv2.resizeWindow('Processed Image', processed_width, processed_height)
            cv2.moveWindow('Processed Image', processed_x, y)
            cv2.imshow('Processed Image', processed_bgr)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"Error displaying images: {str(e)}")