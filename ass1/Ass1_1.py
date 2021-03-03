import numpy as np
import matplotlib.pylab as plt
import pickle

def grad_des(parameter,x,y,iterations,learningrate,degree):
    m = len(x)
    theta = parameter
    po = np.arange(degree+1)
    for j in range(iterations):
        sumlist =np.zeros(degree+1)
        for i in range(m):
            D = (np.ones(degree+1))*x[i]
            sumlist = sumlist + (np.dot(theta,np.power(D,po)) - y[i])*np.power(D,po)
        theta = theta - ((1.0/m)*learningrate*sumlist)
    return theta

def grad_despo4(parameter,x,y,iterations,learningrate,degree):
    m = len(x)
    theta = parameter
    po = np.arange(degree+1)
    for j in range(iterations):
        sumlist =np.zeros(degree+1)
        for i in range(m):
            D = (np.ones(degree+1))*x[i]
            sumlist = sumlist + (2*(np.dot(theta,np.power(D,po)) - y[i])**3)*np.power(D,po)
        theta = theta - ((1.0/m)*learningrate*sumlist)
    return theta

def grad_desmod(parameter,x,y,iterations,learningrate,degree):
    m = len(x)
    theta = parameter
    po = np.arange(degree+1)
    for j in range(iterations):
        sumlist =np.zeros(degree+1)
        for i in range(m):
            D = (np.ones(degree+1))*x[i]
            s=np.sign((np.dot(theta,np.power(D,po)) - y[i]))
            # if(s>1):
            sumlist = sumlist + s*np.power(D,po)
            # if(s<0):
                # sumlist = sumlist - np.power(D,po)
        theta = theta - ((1.0/2*m)*learningrate*sumlist)
    return theta

def splitdata(data,N):
    np.random.shuffle(data)
    m = int(0.8*N)
    training, test = data[:m], data[m:]
    return training,test

def mean_sqr_error(p,test_x,test_y):
    m=len(test_x)
    error=0.0
    for i in range(m):
        error = error + (p(test_x[i])-test_y[i])**2
    return error/(2*m)


if __name__ == '__main__':
    #no of data points
    N=10
    x=np.random.uniform(0,1,N)
    #x=np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
    y=np.sin(2*np.pi*x)
    noise = np.random.normal(0,.3,N)
    y = y + noise
    plt.scatter(x,y)
    data = np.column_stack([x,y])
    training, test = splitdata(data,N)
    train_x = np.array([i[0] for i in training])
    train_y = np.array([i[1] for i in training])
    test_x = np.array([i[0] for i in test])
    test_y = np.array([i[1] for i in test])
    degree =5
    parameter=np.ones(degree+1)*0.5
    parameter1 = grad_desmod(parameter,train_x,train_y,50000,.05,degree)
    p1 = np.poly1d(list(reversed(parameter1)))
    print(p1)
    parameter1 = grad_des(parameter,train_x,train_y,50000,.05,degree)
    p1 = np.poly1d(list(reversed(parameter1)))
    print(p1)
    test_error = mean_sqr_error(p1,test_x,test_y)
    print("Test error is %.2f"%(test_error))
    train_error = mean_sqr_error(p1,train_x,train_y)
    print("Train error is %.2f"%(train_error))
    x1 =np.linspace(0,1,100)
    x1.sort()
    y1 = p1(x1)
    plt.plot(x1, y1)
    plt.show()
