class ScamGrader:

    def easy(self, action, observation, info):
        actual = info.get("actual")
        return 0.9 if action == actual else 0.1

    def medium(self, action, observation, info):
        actual = info.get("actual")
        if action == actual:
            return 0.9
        return 0.5

    def hard(self, action, observation, info):
        actual = info.get("actual")
        return 0.9 if action == actual else 0.1