import json

class SettingsManager:
    def __init__(self, settings_file):
        self.settings_file = settings_file

    def load_settings(self):
        """Load processing settings from JSON or use defaults"""
        try:
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        except:
            settings = {
                'brightness': 1.2,
                'contrast': 1.1,
                'sharpness': 1.0,
                'noise_reduction': True,
                'auto_wb': True,
                'preview_size': (800, 600),
                'save_metadata': True,
                'downsample_threshold': 50000000,
                'power_saving_enabled': True
            }
            self.save_settings(settings)
            return settings

    def save_settings(self, settings):
        """Save current settings to JSON"""
        with open(self.settings_file, 'w') as f:
            json.dump(settings, f, indent=4)