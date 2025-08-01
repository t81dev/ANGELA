class ContextManager:
    """
    Enhanced ContextManager with hierarchical context storage and merging capabilities.
    Tracks user sessions and supports incremental updates.
    """

    def __init__(self):
        self.context = {}

    def update_context(self, user_id, data):
        """
        Update or merge context for a given user.
        """
        if user_id not in self.context:
            self.context[user_id] = {}
        # Merge new data into existing context
        self.context[user_id].update(data)

    def get_context(self, user_id):
        """
        Retrieve the context for a user.
        """
        return self.context.get(user_id, "No prior context found.")

    def clear_context(self, user_id):
        """
        Clear stored context for a user.
        """
        if user_id in self.context:
            del self.context[user_id]

    def list_users(self):
        """
        List all users with stored contexts.
        """
        return list(self.context.keys())
