import cunumeric as cn
import legateboost as lbst

X = cn.random.random((100, 10))
y = cn.random.random(X.shape[0])
model = lbst.LBRegressor(verbose=1, init=None).fit(X, y)
pred = model.predict(X)
print(pred)
