class ScamGrader:
 
    def easy(self, action, observation, info):
        actual = info.get("actual")
        return 0.85 if action == actual else 0.15
 
    def medium(self, action, observation, info):
        actual = info.get("actual")
        return 0.80 if action == actual else 0.40
 
    def hard(self, action, observation, info):
        actual = info.get("actual")
        return 0.75 if action == actual else 0.25
 