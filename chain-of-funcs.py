def add(x):
  def multi(y):
    def div(z):
      return z / 10
    return div(y * 10)
  return multi(x + 10)
result = add(10)
print(f"assert add(10) == 20: {result == 20}")
assert result == 20