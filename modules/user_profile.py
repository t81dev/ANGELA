class UserProfile:
    def __init__(self):
        self.preferences = {"style": "neutral"}

    def get_preferences(self):
        return self.preferences

    def update_preferences(self, new_prefs):
        self.preferences.update(new_prefs)
