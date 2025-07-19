class AlignmentGuard:
    def check(self, user_input):
        # Very basic filter: block dangerous requests
        banned_keywords = ["hack", "virus", "destroy", "harm"]
        if any(keyword in user_input.lower() for keyword in banned_keywords):
            return False
        return True

