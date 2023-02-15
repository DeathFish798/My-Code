import random
import math

class Particle:
    def __init__(self, dim):
        self.position = [random.uniform(-5, 5) for i in range(0, dim)]
        self.velocity = [0.0 for i in range(0, dim)]
        self.best_position = list(self.position)
        self.best_fitness = float("inf")

    def update_velocity(self, global_best_position, w, c1, c2):
        for i in range(0, len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = c1 * r1 * (self.best_position[i] - self.position[i])
            social_velocity = c2 * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = w * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        for i in range(0, len(self.position)):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            if self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

    def evaluate(self, cost_func):
        self.fitness = cost_func(self.position)
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = list(self.position)

class PSO:
    def __init__(self, cost_func, dim, num_particles, max_iter):
        self.cost_func = cost_func
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.particles = [Particle(dim) for i in range(0, num_particles)]
        self.best_particle = None
        self.run_count = 0

    def run(self):
        for i in range(0, self.num_particles):
            self.particles[i].evaluate(self.cost_func)
            if self.best_particle is None or self.particles[i].best_fitness < self.best_particle.best_fitness:
                self.best_particle = self.particles[i]

        for i in range(0, self.max_iter):
            for j in range(0, self.num_particles):
                self.particles[j].update_velocity(self.best_particle.best_position, self.w, self.c1, self.c2)
                self.particles[j].update_position(bounds)
                self.particles[j].evaluate(self.cost_func)
                if self.particles[j].best_fitness < self.best_particle.best_fitness:
                    self.best_particle = self.particles[j]

            self.run_count += 1

        return self.best_particle.best_position

# 定义四元函数
def four_quaternion(x):
    return math.sin(x[0]) * math.sin(x[1]) * math.sin(x[2]) * math.sin(x[3])

# 测试
if __name__ == "__main__":
    dim = 4
    num_particles = 50
    max_iter = 100
    bounds = [(-5, 5) for i in range(0, dim)]
    pso = PSO(four_quaternion, dim, num_particles, max_iter)
    best_position = pso.run()
    best_fitness = four_quaternion(best_position)
    print("最优解：", best_position)
    print("最优解的函数值：", best_fitness)