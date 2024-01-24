import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math

def main():
    number_of_customers, monthly_revenue = read_csv_file()
    visualize_init_data(number_of_customers, monthly_revenue)

    w_init = 0.
    b_init = 0.
    alpha = 0.00055
    num_iters = 10000

    w, b, _, _ = gradient_descent(number_of_customers, monthly_revenue, w_init, b_init, compute_cost, compute_gradient, alpha, num_iters)

    visuale_model(number_of_customers, monthly_revenue, w, b)

    inp = int(input("How many customers in a month: "))

    prediction = predict(inp, w, b)
    print("Aproximately monthly revenue : {0:.2f}".format(prediction))

def read_csv_file():
    df = pd.read_csv("revenue.csv")

    number_of_customers = df["Number_of_Customers"][:2000]
    monthly_revenue = df["Monthly_Revenue"][:2000]

    number_of_customers = np.array(number_of_customers)
    monthly_revenue = np.array(monthly_revenue)

    return number_of_customers, monthly_revenue

def visualize_init_data(number_of_customers, monthly_revenue):
    plt.scatter(number_of_customers, monthly_revenue, marker="x", c="red")
    plt.title("Number of Customers - Monthly Revenue")
    plt.xlabel("Number of Customers")
    plt.ylabel("Monthly Revenue")
    plt.show()

def compute_cost(number_of_customers, monthly_revenue, w, b):
    total_cost = 0
    m = number_of_customers.shape[0]
    cost = 0

    for i in range(m):
        f_wb_i = w * number_of_customers[i] + b
        cost += (f_wb_i - monthly_revenue[i]) ** 2

    total_cost = (1 / (2 * m)) * cost

    return total_cost

def compute_gradient(number_of_customers, monthly_revenue, w, b):
    m = number_of_customers.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb_i = w * number_of_customers[i] + b
        dj_dw_i = (f_wb_i - monthly_revenue[i]) * number_of_customers[i]
        dj_db_i = (f_wb_i - monthly_revenue[i]) * 1

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(number_of_customers, monthly_revenue, w_init, b_init, cost_function, gradient_function, alpha, num_iters):
    J_hist = []
    w_hist = []
    w = copy.deepcopy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(number_of_customers, monthly_revenue, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(number_of_customers, monthly_revenue, w, b)
            J_hist.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            w_hist.append(w)
            print(f"Iteration {i:4}: Cost {float(J_hist[-1]):8.2f}")
    
    return w, b, J_hist, w_hist

def visuale_model(number_of_customers, monthly_revenue, w, b):
    m = number_of_customers.shape[0]
    predicted = np.zeros(m)

    for i in range(m):
        predicted[i] = number_of_customers[i] * w + b

    plt.plot(number_of_customers, predicted, c = "b")
    plt.scatter(number_of_customers, monthly_revenue, marker='x', c='r') 

    plt.title("Number of Customers vs. Monthly Revenue")
    plt.xlabel("Number of Customers")
    plt.ylabel("Monthly Revenue")
    plt.show()

def predict(inp, w, b):
    prediction = inp * w + b

    return prediction

main()