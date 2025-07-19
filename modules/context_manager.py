class ContextManager:
    def __init__(self):
        self.context = {}

    def update_context(self, user_id, data):
        self.context[user_id] = data

    def get_context(self, user_id):
        return self.context.get(user_id, "No prior context found.")
