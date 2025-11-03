from .base import BaseModel


class HumanModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="human", **kwargs)

    def chat(self, messages: list[dict], **kwargs) -> str:
        actions = []
        for i in range(self.n):
            action = input(f"Enter prediction #{i+1}: ")
            actions.append(action)
        return actions
