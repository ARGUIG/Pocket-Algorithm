import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# La classe Pocket
class Pocket :
    def fit(self, X, y, nb_iterations=100 ):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.weights = np.zeros((n_features+1,))
        w = np.zeros((n_features+1,))
        X = np.concatenate([X, np.ones((n_samples,1))], axis=1)

        for i in range(nb_iterations):
            for j in range(n_samples):
                if (y[j]*np.dot(w, X[j, :]) <= 0) :
                    w += y[j]*X[j,:]
                    if self.evaluate(X, y, w) >= self.evaluate(X, y, self.weights) :
                        self.weights = [w[i] for i in range(len(w))]

    def evaluate(self, X, y, weights):
        if not hasattr(self, 'weights'):
            print("le modéle n'a pas encore été entrainé!")
            return
        n_samples = X.shape[0]
        pred_y = np.matmul(X, weights)
        pred_y = np.vectorize(lambda val:1 if val>0 else -1)(pred_y)
        #calcule de precision du modele
        return np.mean(y == pred_y)
    
    def predict(self, X):
        if not hasattr(self, 'weights'):
            print("le modéle n'a pas encore été entrainé!")
            return
        
        n_samples = X.shape[0]
        X = np.concatenate([X, np.ones((n_samples,1))], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val>0 else -1)(y)
        return y
    
    def score(self, X, y):
        pred_y = self.predict(X)
        return np.mean( y == pred_y )
    
#Creation des donnees
N = 300 #Nombre de points par classe
X, y = make_classification(
    n_features = 2,
    n_classes = 2,
    n_samples = 3*N,
    n_redundant = 0,
    n_clusters_per_class = 1,
    random_state = 320,
    class_sep = 0.8,
    flip_y=0.1
)
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolors='k') 
# plt.xlabel('Feature 1') 
# plt.ylabel('Feature 2') 
# plt.title('Données générées aléatoirement') 
# plt.show()

def draw(X, y, a, ax):
    ax.scatter(X[:, 0], X[:, 1], marker='D', c=y, cmap="winter", edgecolors='k')
    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:,0])
    x1_1 = (-a.weights[2] - a.weights[0]*x0_1 )/ a.weights[1]
    x1_2 = (-a.weights[2] - a.weights[0]*x0_2 )/ a.weights[1]
    ax.plot([x0_1,x0_2],[x1_1,x1_2])

y = np.vectorize(lambda val :1 if val>0 else-1)(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=32)
p=Pocket()
p.fit(X_train,y_train)

#plot
fig, (ax_train, ax_test) = plt.subplots(nrows=1, ncols=2, dpi=120, figsize=(8,4))
ax_train.set_title('Train')
ax_test.set_title('Test')
plt.suptitle("Pocket accuracy : " + str(p.score(X_test,y_test)*100)+"%")
draw(X_train,y_train,p,ax_train)
draw(X_test,y_test,p,ax_test)
plt.show()

