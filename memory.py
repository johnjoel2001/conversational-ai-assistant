# memory.py

class Memory:
    """
    A simple in-memory store for a couple of “slots”:
      - user_name
      - favorite_color

    parse_and_store(user_input) will look at your text and try to extract:
      • If you say “My name is Alice”, we store user_name = "Alice".
      • If you say “My favorite color is green”, we store favorite_color = "green".

    to_prompt() will return a short string like:
      "You are talking to Alice. The user likes green."
    """

    def __init__(self):
        self.slots = {}   # holds e.g. { "user_name": "Alice" }

    def get(self, key, default=None):
        return self.slots.get(key, default)

    def set(self, key, value):
        self.slots[key] = value

    def to_prompt(self) -> str:
        """
        Compose a short piece of text from stored slots.
        """
        parts = []
        if "user_name" in self.slots:
            parts.append(f"You are talking to {self.slots['user_name']}.")
        if "favorite_color" in self.slots:
            parts.append(f"The user likes {self.slots['favorite_color']}.")
        return " ".join(parts)

    def parse_and_store(self, user_input: str):
        """
        Naively parse:
          - “My name is X”  → store user_name = X (first word after that phrase).
          - “My favorite color is Y” → store favorite_color = Y (first word after that phrase).
        """
        text = user_input.strip()
        lower = text.lower()

        if lower.startswith("my name is "):
            remainder = text[len("my name is "):].strip()
            name = remainder.split()[0]
            self.set("user_name", name.title())
        elif "my favorite color is " in lower:
            idx = lower.index("my favorite color is ") + len("my favorite color is ")
            remainder = text[idx:].strip()
            color = remainder.split()[0]
            self.set("favorite_color", color.lower())
