import torch
import torch.nn.functional as F

y = torch.tensor([1.0])  # the correct answer

#   x₁ ──[× w₁]──┐
#                ├──(+)── z ──[sigmoid]── a
#           b ───┘

# bias: what to believe having seen nothing yet
# weight: how much to revise that belief per unit of evidence
 
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2]) # weight parameter
b = torch.tensor([0.0])  # bias unit
z = x1 * w1 + b          # net input (logit)
a = torch.sigmoid(z)     # activation

loss = F.binary_cross_entropy(a, y) # measure of error
print(loss)
