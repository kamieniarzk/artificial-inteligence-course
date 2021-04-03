import numpy as np
import math
import sys
import time


'''
12.03.2021

EARIN Exercise 1 - Gradient optimization

Jakub Szumski (295432)
Kacper Kamieniarz (293065)

'''

class Result():
    def __init__(self, x: np.array, c: int, b: np.array, A: np.array, method: bool, step_size: int, max_iter, max_duration, desired_value, verbose=False):
        self.x = x
        self.c = c
        self.b = b
        self.A = A
        self.verbose = verbose
        self.method = method
        self.step_size = step_size
        self.max_iter = max_iter
        self.max_duration = max_duration
        self.desired_value = desired_value

    def function(self, x: np.array) -> np.array:
        return c + self.b.T @ x + x.transpose() @ self.A @ x

    def set_x(self, x: np.array):
        self.x = x

    def gradient(self, x: np.array) -> np.array:
        gradient = np.zeros(shape=(dimension,))
        for row, elem in enumerate(x):
            gradient[row] = self.b[row] + x.T @ A[row] + x.T @ self.A.T[row]
        return gradient

    def newtons_method(self):
        initial_time = time.time()

        H = self.A + self.A.T
        print(f"Initial X: {self.x}")
        print(f"Initial Function value: {self.function(self.x)}")

        for epoch in range(self.max_iter):

            if self.function(self.x) < self.desired_value:
                break

            grad = self.gradient(self.x)
            self.x -= np.linalg.inv(H) @ grad

            if (time.time() - initial_time) * 1000 >= self.max_duration:
                break

            if self.verbose and epoch % math.floor(self.max_iter * 0.1) == 0:
                print(f"Epoch: {epoch}")
                print(f"X: {self.x} X difference: {np.linalg.inv(H) @ grad}")
                print(f"Function value: {self.function(self.x)}")

        return self.x, self.function(x)
        
    def gradient_search(self):
        initial_time = time.time()

        for epoch in range(self.max_iter):
            
            if self.function(x) < self.desired_value:
                break

            grad = self.gradient(self.x)
            self.x -= self.step_size * grad

            if (time.time() - initial_time) * 1000 >= self.max_duration:
                break

            if self.verbose and epoch % math.floor(self.max_iter * 0.01) == 0:
                print(f"Epoch: {epoch}")
                print(f"X values: {x} Gradient Values: {grad}")
                print(f"Function value: {self.function(x)}")
            
        return self.x, self.function(x)

    def optimize(self) -> np.array:
        if not self.method:
            return self.newtons_method()
        else:
            return self.gradient_search()

def is__matrix_positively_defined(A: np.array):
    return np.all(np.linalg.eigvals(A) > 0)


def exit_with_error(msg: str):
    print(msg)
    sys.exit(1)
        


if __name__ == "__main__":
    
    print("Select optimization method")
    print("0 - Newton's Method")
    print("1 - Gradient Descent")
    user_input = int(input("Method: "))

    method = True if user_input == 1 else False

    try:
        dimension = int(input("Specify d - number of dimensions: "))

        if dimension <= 0:
            exit_with_error("Dimensions must be positive")

        c = np.float64(input("Specify c constant: "))
        b = np.array([ np.float64(x) for x in input("Place vector b separated by space: ").split(maxsplit=dimension - 1)])
        A = np.zeros(shape=(dimension, dimension))
        for row in range(dimension):
            A[row] = np.array([np.float64(a) for a in input(f"Place row {row} of matrix A separated by space: ").split(maxsplit=dimension - 1)])

        if not is__matrix_positively_defined(A):
            print("Matrix A is not positively defined")
            print("Function may not find minimum")
        
        print("Select initial x mode")
        print("0 - Uniform Distribution")
        print("1 - Direct Input")
        user_input = int(input("Input mode: "))

        input_mode = True if user_input == 1 else False

        if input_mode is False:
            l = np.float64(input("Set l: "))
            u = np.float64(input("Set u: "))
            x = np.random.uniform(low=l, high=u, size=dimension)
        else:
            x = np.array([ np.float64(i) for i in input("Place vector x separated by space: ").split(maxsplit=dimension - 1)])

        step_size = 0
        if method:
            step_size = np.float64(input("Specify step size: "))
        
        max_iter = int(input("Specify maximal number of iterations: "))
        if max_iter <= 0:
            exit_with_error("Maximal iteration number must be bigger than 0") 
        
        max_duration = int(input("Specify maximal computation time in miliseconds: "))

        if max_duration <= 0:
            exit_with_error("Duration time negative") 

        desired_value = np.float64(input("Specify desired function value: "))

        print("Select mode")
        print("0 - Normal Mode")
        print("1 - Batch Mode")
        user_input = int(input("Mode: "))

        mode = True if user_input == 1 else False

        if mode:
            n = int(input("Specify number of restarts: "))
            if n <= 0:
                exit_with_error("Number of restarts must be positive")


        print("Select verbosity mode")
        print("0 - No verbose")
        print("1 - Verbose")
        user_input = int(input("Mode: "))

        verbosity_mode = True if user_input == 1 else False
        
    except:
        exit_with_error("! Wrong Parameters !")

    if not mode:
        result = Result(x, c, b, A, method, step_size, max_iter, max_duration, desired_value, verbosity_mode)
        x, value = result.optimize()
        print(f"Final x value: {x}")
        print(f"Found minimal function value: {value}")
    else:
        results = []
        single_result = Result(x, c, b, A, method, step_size, max_iter, max_duration, desired_value, verbosity_mode)
        for iteration in range(n):
            if input_mode == 0:
                x = np.random.uniform(low=l, high=u, size=dimension)
            single_result.set_x(x)
            results.append(single_result.optimize()[1])

        print(f"Results for batch mode: {results}")
        print(f"Standard deviation: {np.std(results)}")
        print(f"Mean value: {np.mean(results)}")
    
