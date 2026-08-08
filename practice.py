import sys

sys_= {
    "hope": "yonsei",
    "primary": "snu",
    "secondary" : "ku",
    "major" : "electrical enginnering",
    "club" : "car",
    "orientation" :"money"
}

for k,v in sys_.items():
    setattr(sys, k, v)

print(sys.hope)
print(sys.primary)
print(sys.secondary)
print(sys.major)
print(sys.club)
print(sys.orientation)