from mxnet import np, npx

npx.set_np()
x = np.arange(12)

fair_probs = [1.0 / 6]*6
print(fair_probs)
print(np.random.multinomial(10, fair_probs))