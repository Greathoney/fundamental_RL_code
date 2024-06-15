import numpy as np

class RaceCar:    
    # 0: do nothing, 1: up, 2: up+right, 3: right, 4: down+right, 5: down, 6: down+left, 7: left, 8: up+left
    action_table = np.array([[0, 0], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]])

    def __init__(self, map_path):
    
        with open(map_path, 'r') as file:
            lines = file.readlines()

        data = [list(line.strip()) for line in lines]
        # 0: wall, 1: road, 2: start, 3: finish
        self.track = np.array(data, dtype=int)

        # find all start positions
        self.start_positions = np.argwhere(self.track == 2)
        self.start_positions = [tuple(pos) for pos in self.start_positions]

        self.end_positions = np.argwhere(self.track == 3)
        self.end_positions = [tuple(pos) for pos in self.end_positions]
        self.end_position_x = self.end_positions[0][1]
        self.end_position_y = (self.end_positions[0][0], self.end_positions[-1][0])

    def reset(self, noise=0.1):
        self.pos = self.start_positions[np.random.randint(len(self.start_positions))]
        self.vel = np.array([0, 0])
        self.noise = noise

        return (*self.pos, *self.vel)
    
    def is_stopped(self, action):
        return np.array_equal(self.vel + self.action_table[action], [0, 0])

    def step(self, action):
        if self.noise < np.random.rand():
            self.vel += self.action_table[action]
            # limit the velocity
            self.vel = np.clip(self.vel, -4, 4)

        new_pos = self.pos + self.vel

        # check if the car reaches the goal
        if self.pos[1] < self.end_position_x <= new_pos[1]:
            slope = (new_pos[0] - self.pos[0]) / (new_pos[1] - self.pos[1])
            y = slope * (self.end_position_x - self.pos[1]) + self.pos[0]
            if self.end_position_y[0]-0.5 <= y <= self.end_position_y[1]+0.5:
                return (*new_pos, *self.vel), 0, True
        
        # check if the car is out of the track
        if new_pos[0] < 0 or new_pos[0] >= self.track.shape[0] or new_pos[1] < 0 or new_pos[1] >= self.track.shape[1] or self.track[new_pos[0], new_pos[1]] == 0:
            self.reset(self.noise)
            return (*self.pos, *self.vel), -1, False

        self.pos = new_pos
        return (*self.pos, *self.vel), -1, False
    
    def render(self):
        track = np.copy(self.track)
        track[self.pos[0], self.pos[1]] = 4
        for line in track:
            print(''.join([str(i) for i in line]))
        print()

if __name__ == '__main__':
    env = RaceCar('map1.txt')
    done = False
    while not done:
        env.render()
        action = int(input("Enter action: "))
        (state, vel), reward, done = env.step(action)
