class ScamGrader:

    def easy(self, action, observation, info):
        actual = info.get("actual")
        return 1.0 if action == actual else 0.0

    def medium(self, action, observation, info):
        actual = info.get("actual")
        if action == actual:
            return 1.0
        return 0.5

    def hard(self, action, observation, info):
        actual = info.get("actual")
        return 1.0 if action == actual else 0.0