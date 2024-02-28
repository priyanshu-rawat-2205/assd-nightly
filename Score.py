class Score:

    FRAME_IN_CM = 17.8
    FRAME_IN_PX = 512
    RADIUS = 0.6
    WIDTH = 0.8

    PX2CM = FRAME_IN_CM/FRAME_IN_PX

    def px2cm(self, dist):
        return dist * self.PX2CM
    
    def finalScore(self, distance):

        dist = self.px2cm(distance)

        if dist == 0:
            return 10.9
        elif dist <= 0.6:
            decimal = (self.RADIUS - dist)//0.06
            score = 10 + (decimal/10)
            return score
        elif dist > 0.6 and dist <= 1.4:
            decimal = (self.RADIUS + self.WIDTH - dist)//0.08
            score = 9 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 1)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 1) - dist)//0.08
            score = 8 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 2)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 2) - dist)//0.08
            score = 7 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 3)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 3) - dist)//0.08
            score = 6 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 4)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 4) - dist)//0.08
            score = 5 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 5)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 5) - dist)//0.08
            score = 4 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 6)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 6) - dist)//0.08
            score = 3 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 7)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 7) - dist)//0.08
            score = 2 + (decimal/10)
            return score
        elif dist <= ((self.RADIUS + self.WIDTH) + (self.WIDTH * 8)):
            decimal = ((self.RADIUS + self.WIDTH) + (self.WIDTH * 8) - dist)//0.08
            score = 1 + (decimal/10)
            return score
        elif dist > ((self.RADIUS + self.WIDTH) + (self.WIDTH * 8)):
            return 0
        
