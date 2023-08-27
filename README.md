# 手鼠：未來操控的可能性
**滑鼠** 在現代科技發展的浪潮中，滑鼠作為電腦使用者界面不可或缺的一部分，已經成為了人類與電腦互動的途徑之一。滑鼠不僅能夠實定位，還能夠進行點擊、選擇等重要的操作，無疑在人機交互領域扮演著不可或缺的角色。然而，隨著科技的進步，許多創新的操控方式正在逐漸取代傳統的滑鼠，例如搖桿等裝置。  
  
然而，在我們腦海中，是否曾經浮現過一個奇幻的畫面——就像鋼鐵人電影中的場景一樣，有著一個神奇的裝置，這個裝置不需要滑鼠，僅僅通過手部動作就能實現電腦的所有操作。這種想法或許對大多數人來說只是科幻，但是科技的發展往往源於人類對未來的不斷探索和想像。  
  
這樣的裝置將不僅僅是一個科技的突破，更是人類智慧的結晶。想像一下，用手指的輕點來打開應用程序，用手掌的平移來進行文件的拖拽，用手的姿勢來調整音量和亮度。這將不僅僅是一種操控方式，更是一種藝術，一種與電腦交流的新語言。  

## 版本紀錄  

> **v.1**  
 ```
新增兩種動作 "點擊"(握拳) "移動"(手掌移動)
使用了OpenCv獲得鏡頭畫面，並使用Mediapipe對手部姿勢進行判斷
以Mediapipe的手掌中心(0號節點)為基準點，透過AutoGui將滑鼠移動到螢幕上的相對位置
並以 "食指指尖(8號節點)與食指第一指節(5號節點)的距離" 和 "食指第二指節(6號節點)與食指第一指節(5號節點)的距離" 進行對比，作為判斷是否握拳的依據
 ```
  
> **v.2**
```
改變了握拳的判斷方式
以拇指指尖(4號節點)與無名指第二指節(15號節點)的距離為判斷依據
```
  
> **v.3**
```
改變了動作的判斷方式
採用Google Teachable Machine 訓練模型
```
  
> **v.4**
```
改變了動作的判斷方式
採用數學算法計算手指角度
基於https://blog.csdn.net/weixin_45930948/article/details/115444916進行修改
新增滾輪動作(食指中指併攏左右傾斜)
```
  
> **v.5**
```
使用多線程處裡不同任務以減少單線程的負載
```

> **v.6**
```

```
