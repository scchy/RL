import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt

def random_action(env):
    return np.random.uniform(env.action_space.low, env.action_space.high)

env_name = 'CarRacing-v2'
env = gym.make(env_name)
s, _ = env.reset()
a=random_action(env)
n_state, reward, done, info1, info2 = env.step(a)
img = cv2.Canny(n_state, 170, 250)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
axes[0].imshow(n_state)
axes[0].set_title('n_state img')
axes[1].imshow(img)
axes[1].set_title('Canny img')
axes[2].imshow(img[:80, :80])
axes[2].set_title('Canny img[:80, :80]')
plt.show()

img_a = img[60:80, :80]
x, y, w, h = cv2.boundingRect(img_a)
cv2.rectangle(img_a, (x, y), (x + w, y + h), (255, 0, 0), 1)

plt.imshow(img_a)
plt.show()

# detector car in street , else rewards = -100


hsv = cv2.cvtColor(n_state, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (36, 25, 25), (80, 255, 255))
imask = mask<=0
no_green = np.zeros_like(n_state, np.uint8)
no_green[imask] = n_state[imask]
no_green = no_green[:80, :80]
plt.imshow(no_green)
plt.show()

img = cv2.Canny(no_green, 170, 250)
plt.imshow(img)
plt.show()
# plt.imshow(s[..., 1])
# plt.show()

img = cv2.Canny(n_state, 170, 250)
img = img[60:80, :80]
ret,thresh = cv2.threshold(img, 50, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)
print("Number of contours detected:", len(contours))
for cnt in contours:
    print(cnt.shape)
    x1,y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    if len(approx) < 4:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    ratio= float(w)/h




img[img==0].shape, img[img>0].shape, 80*80, img[img==0].shape[0] + img[img>0].shape[0]
