### IoU

```
def IOU( box1, box2 ):
    """
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    """
    width1, height1 = abs(box1[2] - box1[0]), abs(box1[1] - box1[3])
    width2, height2 = abs(box2[2] - box2[0]), abs(box2[1] - box2[3])
    
    # 并区域的x，y的最大最小值
    x_max, x_min = max(box1[0],box1[2],box2[0],box2[2]), min(box1[0],box1[2],box2[0],box2[2])
    y_max, y_min = max(box1[1],box1[3],box2[1],box2[3]), min(box1[1],box1[3],box2[1],box2[3])
    
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height # 交集的面积
        box1_area, box2_area = width1 * height1, width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio
    
box1 = [1,3,4,1]
box2 = [2,4,5,2]
print(IOU(box1,box2))
```

### NMS [(出处)](https://blog.csdn.net/a1103688841/article/details/89711120)
```
def nms(self, bboxes, scores, thresh=0.5):
    x1, y1 = bboxes[:,0], bboxes[:,1]
    x2, y2 = bboxes[:,2], bboxes[:,3]
    areas = (y2-y1+1)*(x2-x1+1)
    scores = bboxes[:,4]
    
    keep = []
    index = scores.argsort()[::-1]
    
    while index.size > 0:
        i = index[0] # 取出第一个方框进行和其他方框比对，看看有没有合并，第一个总是最大的
        
        keep.append(i) # keep保留的是索引值，不是分数
        # 计算交集的左上角和右下角
        x_lt, y_lt = np.maximum(x1[i], x1[index[1:]]), np.maximum(y1[i], y1[index[1:]])
        x_rb, y_rb = np.minimum(x2[i], x2[index[1:]]), np.minimum(y2[i], y2[index[1:]])
        
        # 如果两个方框相交，x_rb-x_lt和y_rb-y_lt是正的，如果两个方框不相交，x_rb-x_lt和y_rb-y_lt是负的，我们把不相交的W和H设为0.
        w, h = np.maximum(0, x_rb-x_lt+1), np.maximum(0, y_rb-y_lt+1)
        overlaps = w * h
        IoUs = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        # 接下来是合并重叠度最大的方框，也就是合IoUs中值大于thresh的方框，合并这些方框只保留下分数最高的。经过排序当前我们操作的方框就是分数最高的，所以剔除其他和当前重叠度最高的方框
        idx = np.where(IoUs <= thresh)[0]
        
        #把留下来框在进行NMS操作，留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框，每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx+1]
    
    return keep
  
```

## IoU C++
```
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

class bbox
{
public:
    bbox();
    bbox(int lt_x, int lt_y, int br_x, int br_y);
    friend float IoU(bbox b1, bbox b2);
    ~bbox();
private:
    int ltx;
    int lty;
    int brx;
    int bry;
};

bbox::bbox(int lt_x, int lt_y, int br_x, int br_y){
    ltx = lt_x;
    lty = lt_y;
    brx = br_x;
    bry = br_y;
}

bbox::~bbox(){}

float IoU(bbox b1, bbox b2) {
    float iou_ratio;
    // 两个框的宽和高
    int width1 = abs(b1.brx - b1.ltx);
    int height1 = abs(b1.bry - b1.lty);
    int width2 = abs(b2.brx - b2.ltx);
    int height2 = abs(b2.bry - b2.lty);

    // 并区域x,y的最大值
    int x_max = max(max(b1.ltx, b1.brx), max(b2.ltx, b2.brx));
    int x_min = min(min(b1.ltx, b1.brx), min(b2.ltx, b2.brx));
    int y_max = max(max(b1.lty, b1.bry), max(b2.lty, b2.bry));
    int y_min = min(min(b1.lty, b1.bry), min(b2.lty, b2.bry));

    // 交区域的宽和高
    int iou_width = x_min + width1 + width2 - x_max;
    int iou_height = y_min + height1 + height2 - y_max;

    if (iou_width <= 0 || iou_height <= 0)  iou_ratio = 0;
    else {
        int iou_area = iou_width * iou_height;
        int area1 = width1 * height1;
        int area2 = width2 * height2;
        iou_ratio = (float)iou_area / (float)(area1 + area2 - iou_area);
    }
    return iou_ratio;
}

int main()
{
    bbox box1(1,5,3,2), box2(2,8,5,3);
    float iu = IoU(box1, box2);
    cout << iu;
    return 0;
}
```
# 《数据结构与算法》
## 排序
[912. 排序数组](https://leetcode-cn.com/problems/sort-an-array/)

给你一个整数数组 nums，请你将该数组升序排列。

***示例***

**输入**: nums = [5,2,3,1]

**输出**: [1,2,3,5]

### 快速排序 ***时间O(nlogn)，空间O(logn)，不稳定***

```
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        int n = nums.size();
        quickSort(nums, 0, n-1);
        return nums;
    }

    void quickSort(vector<int>& nums, int start, int end) {
        if (start < end) {
            int mid = partition(nums, start, end);
            quickSort(nums, start, mid - 1);
            quickSort(nums, mid + 1, end);
        }
    }

    int partition(vector<int>& nums, int start, int end) {
        int current = nums[start];
        while (start < end) {
            while (start < end && nums[end] >= current) --end;
            nums[start] = nums[end];
            while (start < end && nums[start] <= current) ++start;
            nums[end] = nums[start];
        }
        nums[start] = current;
        return start;
    }
};
```

### 归并排序 ***时间O(nlogn)，空间O(n)，稳定***
```
class Solution{
public:
    vector<int> sortArray(vector<int>& nums) {
        int n=nums.size();
        if(n<=1)return nums;
        vector<int> temp(n);
        mergeSort(nums,0,n-1,temp);
        return nums;
    }
    void mergeSort(vector<int>& nums,int first,int last,vector<int>& temp)
    {
        if(first<last)
        {
            int mid=(first+last)/2;
            mergeSort(nums,first,mid,temp);
            mergeSort(nums,mid+1,last,temp);
            merge(nums,temp,first,mid,last);
        }
    }
    void merge(vector<int>& nums,vector<int>& temp,int first,int mid,int last)
    {
        int index1=first,index2=mid+1;
        int t=first;
        while(index1<=mid && index2<=last)
        {
            if(nums[index1]<=nums[index2])temp[t++]=nums[index1++];
            else temp[t++]=nums[index2++];
        }
        while(index1<=mid) temp[t++]=nums[index1++];
        while(index2<=last)temp[t++]=nums[index2++];
        for(t=first;t<=last;++t)
        {
            nums[t]=temp[t];
        }
    }
};
```

### 堆排序 ***时间O(nlogn)，空间O(1)，不稳定***
```
class Solution{
public:
    vector<int> sortArray(vector<int>& nums) {
        int n=nums.size();
        if(n<=1)return nums;
        return heapSort(nums,n);
    }
    vector<int>heapSort(vector<int>& nums,int n)
    {
        for(int i=n/2-1;i>=0;--i)
        siftDown(nums,i,n-1);
        for(int i=n-1;i>=1;--i)
        {
            swap(nums[0],nums[i]);
            --n;
            siftDown(nums,0,n-1);
        }
        return nums;
    }
    void siftDown(vector<int>& nums,int start,int end)
    {
        int j=2*start+1; 
        while(j<=end)
        {
            if(j<end && nums[j]<nums[j+1])++j;
            if(nums[start]>=nums[j])break;
            else
            {
                swap(nums[start],nums[j]);
                start=j;j=2*j+1;
            }
        }
    }
};
```
### 冒泡排序
```
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        int n=nums.size();
        for(int i=0;i<n;i++)
        {
            bool flag=false;
            for(int j=n-1;j>i;j--)
            {
                if(nums[j]<nums[j-1])
                {
                    swap(nums[j],nums[j-1]);
                    flag=true;
                }
            }
            if(flag==false) return nums;
        }
        return nums;
    }
};
```
### 插入排序
```
class Solution{
public:
    vector<int> sortArray(vector<int>& nums) {
        int n=nums.size();
        for(int i=1;i<n;i++)
        {
            int cur=nums[i];
            int index=i-1;
            while(index>=0 && cur<nums[index])
            {
                nums[index+1]=nums[index];
                index--;
            }
            nums[index+1]=cur;
        }
        return nums;
    }
};
```
### 选择排序
```
class Solution{
public:
    vector<int> sortArray(vector<int>& nums) {
        int n=nums.size();
        if(n<=1)return nums;
        for(int i=0;i<n;i++)
        {
            int loc=i;
            for(int j=i;j<n;j++)
            {
                if(nums[j]<nums[loc])
                {
                    loc=j;
                }
            }
            swap(nums[loc],nums[i]);
        }
        return nums;
    }
};

```

## 字符串

### KMP算法（字符串匹配）

#### 28.[实现strStr()](https://leetcode-cn.com/problems/implement-strstr/)

给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。

***示例1***

**输入**: haystack = "hello", needle = "ll"

**输出**: 2

***示例2***

**输入**: haystack = "aaaaa", needle = "bba"

**输出**: -1

```
class Solution {
public:
    int strStr(string haystack, string needle) {
        int n = haystack.size(), m = needle.size();
        if (m == 0) {
            return 0;
        }
        vector<int> pi(m);
        for (int i = 1, j = 0; i < m; i++) {
            while (j > 0 && needle[i] != needle[j]) {
                j = pi[j - 1];
            }
            if (needle[i] == needle[j]) {
                j++;
            }
            pi[i] = j;
        }
        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && haystack[i] != needle[j]) {
                j = pi[j - 1];
            }
            if (haystack[i] == needle[j]) {
                j++;
            }
            if (j == m) {
                return i - m + 1;
            }
        }
        return -1;
    }
};
```

## 图

### 最小生成树：普利姆(Prim)算法，克鲁斯卡尔(Kruskal)算法

#### [1135. 最低成本连通所有城市](https://leetcode-cn.com/problems/connecting-cities-with-minimum-cost/)

想象一下你是个城市基建规划者，地图上有 N 座城市，它们按以 1 到 N 的次序编号。

给你一些可连接的选项 conections，其中每个选项 conections[i] = [city1, city2, cost] 表示将城市 city1 和城市 city2 连接所要的成本。（连接是双向的，也就是说城市 city1 和城市 city2 相连也同样意味着城市 city2 和城市 city1 相连）。

返回使得每对城市间都存在将它们连接在一起的连通路径（可能长度为 1 的）最小成本。该最小成本应该是所用全部连接代价的综合。如果根据已知条件无法完成该项任务，则请你返回 -1。

***示例***

**输入**: N = 3, conections = [[1,2,5],[1,3,6],[2,3,1]]

**输出**: 6

**解释**: 选出任意 2 条边都可以连接所有城市，我们从中选取成本最小的 2 条。


### 最短路径：迪杰斯特拉(Dijkstra)算法

#### [743. 网络延迟时间](https://leetcode-cn.com/problems/network-delay-time/)

有 n 个网络节点，标记为 1 到 n。

给你一个列表 times，表示信号经过有向边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点， wi 是一个信号从源节点传递到目标节点的时间。

现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。

***示例***

**输入**: 

times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2

**输出**: 2
```
#define MAXVALUE 0x3f3f3f3f

class Solution {
public:
    vector<unordered_map<int, int>> mp;
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // 建图 - 邻接表
        mp.resize(n + 1);
        for (auto& edg : times) {
            mp[edg[0]][edg[1]] = edg[2];
        }
        // 记录结点最早收到信号的时间
        vector<int> r(n + 1, MAXVALUE);
        r[k] = 0;

        // 记录已经找到最短路的结点
        unordered_set<int> s;

        while (true) {
            // 待查集中查找最近结点
            int cur = -1, tim = MAXVALUE;
            for (int i = 1; i <= n; ++i) {
                if (r[i] < tim && s.find(i) == s.end()) {
                    cur = i;
                    tim = r[i];
                }
            }

            if (cur == -1) break;

            // 将最近结点加入已查集合并更新
            s.emplace(cur);
            for (auto& edg : mp[cur]) {
                r[edg.first] = min(r[edg.first], edg.second + tim);
            }
        }

        int minT = -1;
        for (int i = 1; i <= n; ++i)
            minT = max(minT, r[i]);
        return minT == MAXVALUE ? -1 : minT;
    }
};
```
# 动态规划————01背包问题

## [SEU-1019](http://47.99.179.148/problem.php?id=1019)可以不装满

有一个容量为C(C<=100)的背包以及N(N<=500)颗宝石，第i颗宝石大小为si，价值为vi。由于条件限制，你手边只有这个背包可作为你搬运宝石的唯一工具。现在你想知道在最多可以带走多大价值的宝石。

***示例***

**输入**: 第一行输入M(M<=10)表示有M组数据。每组数据第一行输入N、C，表示宝石数目以及背包容量；接下来一行输入N组(si,vi), si和vi均为整数，表示每颗宝石的大小和价值。

Sample Input

3

3 10

1 3 2 5 7 2

3 10

1 3 2 5 6 2

5 10

5 6 5 7 2 8 8 1 5 9

**输出**: 输出M行正整数，第i行表示第i组数据可以带走的宝石的最大代价, 背包可不用装满。

10

10

17

```
#include<cstdio>
#include<string.h>
using namespace std;
int f[5001], w[501], v[501];
int ans[11];
int main()
{
    int M;
    scanf("%d", &M);
    for(int j = 0; j < M; j ++)
    {
        int C, N;
        scanf("%d %d", &N, &C);
        memset(f, 0, sizeof(f));
        memset(w, 0, sizeof(w));
        memset(v, 0, sizeof(v));
        for(int i = 0; i < N; ++i)
            scanf("%d %d", &w[i], &v[i]);
        for(int i = 0; i < N; ++i)
            for(int k = C; k >= w[i]; --k)
                if(f[k] < f[k - w[i]] + v[i])
                    f[k] = f[k - w[i]] + v[i];
        ans[j] = f[C];
    }
    for(int j = 0; j < M; j ++)
    {
        if(j == M - 1) printf("%d", ans[j]);
        else printf("%d\n", ans[j]);
    }
    return 0;
}
```

## [SEU-1018](http://47.99.179.148/problem.php?id=1018)必须装满

有一个容量为C(C<=100)的奇怪背包，这个背包可以被带走仅当它恰好被装满。现在你手边有N(N<=500)颗宝石，第i颗宝石大小为si，价值为vi。由于条件限制，你手边只有这个奇怪的背包可作为你搬运宝石的唯一工具。现在你想知道在这样的条件下你最多可以带走多大利润的宝石。

***示例***

**输入**: 第一行输入M(M<=10)表示有M组数据。每组数据第一行输入N、C，表示宝石数目以及背包容量；接下来一行输入N组(si,vi), si和vi均为整数，表示每颗宝石的大小和价值。

Sample Input

3

3 10

1 3 2 5 7 2

3 10

1 3 2 5 6 2

5 10

5 6 5 7 2 8 8 1 5 9

**输出**: 输出M行正整数，第i行表示第i组数据可以带走的宝石的最大代价, 背包可被带走仅当它恰好被装满。

10

0

17
```
#include <iostream>
#include <string.h>
using namespace std;
int s[501],v[501],dp[101];

int main()
{
    int M;
    cin>>M;
    int inf = 9999;
    for(int i=0; i<M; i++){
        int N,C;
        cin>>N>>C;
        for(int i=1; i<=N; i++)
            cin>>s[i]>>v[i];
        dp[0] = 0;
        for(int i = 1; i <= C; i++)
            dp[i] = -inf;
        for(int i=1; i<=N; i++){
            for(int j=C; j>=1; j--){
                if(j >= s[i])
                    dp[j] = max(dp[j],dp[j-s[i]]+v[i]);
                if(dp[j] < 0)
                    dp[j] = -inf;
            }
        }
        if (dp[C] > 0)
            cout<<dp[C]<<endl;
        else
            cout<<0<<endl;
        memset(dp,0,sizeof(dp));
        memset(s,0,sizeof(s));
        memset(v,0,sizeof(v));
    }
    return 0;
}
```

# 动态规划————最长公共子序列

## 1143. [最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

给定两个字符串 text1 和 text2，返回这两个字符串的最长 公共子序列 的长度。如果不存在 公共子序列 ，返回 0 。

一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。


***示例***

**输入**: text1 = "abcde", text2 = "ace" 

**输出**: 3

**解释**: 最长公共子序列是 "ace" ，它的长度为 3 
```
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.length(), n = text2.length();
        vector<vector<int>> dp(m + 1, vector<int>(n + 1));
        for (int i = 1; i <= m; i++) {
            char c1 = text1.at(i - 1);
            for (int j = 1; j <= n; j++) {
                char c2 = text2.at(j - 1);
                if (c1 == c2) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
};

```
# 贪心————最长连续递增序列

## 674.[最长连续递增序列](https://leetcode-cn.com/problems/longest-continuous-increasing-subsequence/)

给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。

连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l <= i < r，都有 nums[i] < nums[i + 1] ，那么子序列 [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。

***示例***

**输入**: nums = [1,3,5,4,7]

**输出**: 3

**解释**：最长连续递增序列是 [1,3,5], 长度为3。尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 

```
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int ans = 0;
        int n = nums.size();
        int start = 0;
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] <= nums[i - 1]) {
                start = i;
            }
            ans = max(ans, i - start + 1);
        }
        return ans;
    }
};
```

# 链表

## 6. [从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

***示例***

**输入**: 

n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]

src = 0, dst = 2, k = 1

**输出**: 200

```
class Solution {
public:
    vector<int> reversePrint(ListNode* head) {
        vector<int> result;
        ListNode *p = head;
        while(p != NULL){
            result.push_back(p->val);
            p = p -> next;
        }
        //rbegin就是逆序
        return vector<int>(result.rbegin(), result.rend());
    }
};
```
```
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        result = []
        while head:
            result.append(head.val)
            head = head.next
        return result[::-1]
```

## 18. [删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点

***示例***

**输入**: head = [4,5,1,9], val = 5

**输出**: [4,1,9]

**解释**: 给定你链表中值为 5 *的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

```
class Solution {
public:
    ListNode* deleteNode(ListNode* head, int val) {
        ListNode *pre = head, *cur = head->next;
        if (head->val == val) return head->next;
        while (cur) {
            if (cur->val == val) {
                pre->next = cur->next;
                cur->next = NULL;
                break;
            }
            pre = cur;
            cur = cur->next;
        }
        return head;
    }
};
```
```
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        pre, cur = head, head.next
        if head.val == val:
            return head.next
        while cur:
            if cur.val == val:
                pre.next = cur.next
                cur.next = None
            pre = cur
            cur = cur.next
        return head
```

## 22. 链表中倒数第k个节点
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

***示例***

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        ListNode *fast = head, *slow = head;
        for (int i = 0; i < k; i++)
            fast = fast->next;
        while (fast) {
            fast = fast->next;
            slow = slow->next;
        }
        return slow;
    }
};
```
```
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast, slow = head, head
        for i in range(0, k):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
 ```
## 24. 反转链表

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

***示例***

**输入**: 1->2->3->4->5->NULL

**输出**: 5->4->3->2->1->NULL
 ```
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == NULL) return head;
        ListNode *cur = NULL, *pre = head, *tmp = head->next;
        while (tmp) {
            pre->next = cur;
            cur = pre;
            pre = tmp;
            tmp = tmp->next;
        }
        pre->next = cur;
        return pre;
    }
};
 ```
 ```
% 完整程序的反转链表
#include <iostream>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
};

ListNode* reverseList(ListNode* head) {
    if (head == NULL) return head;
    ListNode *cur = NULL, *pre = head, *tmp = head->next;
    while (tmp) {
        pre->next = cur;
        cur = pre;
        pre = tmp;
        tmp = tmp->next;
    }
    pre->next = cur;
    return pre;
}

int main() {
    ListNode *head = new ListNode;
    head->val = 1;
    head->next = NULL;
    ListNode *second = new ListNode;
    second->val = 2;
    second->next = NULL;
    head->next = second;
    cout << head->val << " " << second->val << endl;
    ListNode *head1 = new ListNode;
    head1 = reverseList(head);
    cout << head1->val << " " << head1->next->val;
    return 0;
}

 ```
 ```
 class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None: return None
        cur, pre, tmp = None, head, head.next
        while tmp:
            pre.next = cur
            cur = pre
            pre = tmp
            tmp = tmp.next
        pre.next = cur
        return pre
 ```
## 25. 合并两个排序的链表

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

***示例***

**输入**: 1->2->4, 1->3->4

**输出**: 1->1->2->3->4->4
 ```
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == NULL) {
            return l2;
        }
        if(l2 == NULL) {
            return l1;
        }

        if(l1->val < l2->val) {
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2->next = mergeTwoLists(l1, l2->next);
            return l2;
        }
    }
};
 ```
 ```
 class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 == None:
            return l2
        if l2 == None:
            return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
 ```
## 52. 两个链表的第一个公共节点

输入两个链表，找出它们的第一个公共节点。

***示例***

**输入**: listA = [0,9,1,2,4], listB = [3,2,4]

**输出**: Reference of the node with value = 2

![avatar](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png)

 ```
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *curA = headA, *curB = headB;
        while (curA != curB) {
            if (curA != NULL) 
                curA = curA->next;
            else
                curA = headB;
            if (curB != NULL)
                curB = curB->next;
            else    
                curB = headA;
        }
        return curA;
    }
};
 ```
 ```
 class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        curA, curB = headA, headB
        while curA != curB:
            if curA != None:
                curA = curA.next
            else:
                curA = headB
            if curB != None:
                curB = curB.next
            else:
                curB = headA
        return curA
  ```
# 动态规划
## 10-I. 斐波那契数列
写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1

F(N) = F(N - 1) + F(N - 2), 其中 N > 1.

斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

~~答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。~~

***示例***

**输入**: n = 2

**输出**: 1

  ```
class Solution {
public:
    int fib(int n) {
        int result[2] = {0,1};
        if (n < 2)
            return result[n];

        int fibN_minus_1 = 1;
        int fibN_minus_2 = 0;
        int fibN = 0;
        for (int i = 2; i <= n; ++i) {
            fibN = (fibN_minus_1 + fibN_minus_2) % 1000000007;
            fibN_minus_2 = fibN_minus_1;
            fibN_minus_1 = fibN;
        }
        return fibN;
    }
};
  ```
  ```
class Solution:
    def fib(self, n: int) -> int:
        result = [0, 1]
        if n < 2:
            return result[n]
        fibN_minus1, fibN_minus2 = 1, 0
        fibN = 0
        for i in range(2, n+1):
            fibN = (fibN_minus1 + fibN_minus2) % 1000000007
            fibN_minus2 = fibN_minus1
            fibN_minus1 = fibN           
        return fibN
   ```
## 10-II. [青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)   

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

~~答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。~~

***示例***

**输入**: n = 7

**输出**: 21

很像斐波那契数列

```
class Solution {
public:
    int numWays(int n) {
        int base[2] = {1,1};
        int ans_1 = 1, ans_2 = 1, ans = 0;
        if (n == 0 || n == 1) return 1;
        else
        for (int i = 2; i <= n; i++) {
            ans = (ans_1 + ans_2) % 1000000007;
            ans_2 = ans_1;
            ans_1 = ans;
        }
        return ans;
    }
};
```
```
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

## 14-I. [剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]\*k[1]\*...\*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

***示例***

**输入**: 10

**输出**: 36

**解释**: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36

   ```
public:
    int cuttingRope(int n) {
        if (n < 2) return 0;
        if (n == 2) return 1;
        if (n == 3) return 2;

        int *products = new int[n + 1]; //存放每个长度的最优解
        products[0] = 0;
        products[1] = 1;
        products[2] = 2;
        products[3] = 3;

        int max = 0;
        for (int i = 4; i <= n; i++) {
            max = 0;
            for (int j = 1; j <= i/2; j++) {
                int product = products[j] * products[i - j];
                if (max < product)
                    max = product;
                products[i] = max;
            }
        }
        max = products[n];
        return max;
    }
};
   ```

## 42. [连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)
输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

***示例***

**输入**: nums = [-2,1,-3,4,-1,2,1,-5,4]

**输出**: 6

**解释**: 连续子数组 [4,-1,2,1] 的和最大，为 6。
   ```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i - 1] > 0)
                nums[i] += nums[i - 1];
            ans = max(ans, nums[i]);
        }
        return ans;
    }
};
   ```
   ```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res = nums[0]
        for i in range(1,len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
            res = max(res, nums[i])
        return res
   ```   
## 63. [股票的最大利润-解决方案](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)
假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

***示例1***

**输入**: [7,1,5,3,6,4]

**输出**: 5

**解释**: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。

***示例2***

**输入**: [7,6,4,3,1]

**输出**: 0

**解释**: 在这种情况下, 没有交易完成, 所以最大利润为 0。

```   
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minCost = 10000000, maxBenefit = 0;
        for (int price: prices) {
            minCost = min(minCost, price);
            maxBenefit = max(maxBenefit, price - minCost);
        }
        return maxBenefit;
    }
};
```   
```   
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minCost, maxBenefit = 10000000, 0
        for price in prices:
            minCost = min(minCost, price)
            maxBenefit = max(maxBenefit, price - minCost)
        return maxBenefit
```   
# 哈希表
## 3. [数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)
***示例***

**输入**: [2, 3, 1, 0, 2, 5, 3]

**输出**: 2或3

```   
class Solution {
public:
    int findRepeatNumber(vector<int>& nums) {
        for (int i = 0; i < nums.size(); i++) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]])
                    return nums[i];
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }
        return -1;
    }
};
```   
```   
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        for i in range(0, len(nums)):
            while nums[i] != i:
                if nums[i] == nums[nums[i]]:
                    return nums[i]
                temp = nums[i]
                nums[i] = nums[temp]
                nums[temp] = temp
        return -1
```   
## 50. [第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

***示例***

**输入**: s = "abaccdeff"

**输出**: 'b'

C++里有一个 **unorder_map<int, int>** 函数， python使用**字典**
```  
class Solution {
public:
    char firstUniqChar(string s) {
        unordered_map<int, int> frequency;
        for(char a : s)
            ++frequency[a];        
        for(int i = 0; i < s.size(); i++){
            if(frequency[s[i]] == 1)
                return s[i];     
        }
        return ' ';
    }
};
```  
```  
class Solution:
    def firstUniqChar(self, s: str) -> str:
        dic = {}
        for c in s:
            dic[c] = not c in dic
        for k, v in dic.items():
            if v: return k
        return ' '
```  
## 48. [无重复字符的最长子串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

***示例1***

**输入**: "abcabcbb"

**输出**: 3 

**解释**: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

***示例2***

**输入**: "bbbbb"

**输出**: 1 

**解释**: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

# 树

## 7. 重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

***示例***

![btree](https://assets.leetcode.com/uploads/2021/02/19/tree.jpg)

**输入**: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
**输出**: [3,9,20,null,null,15,7]

## 28. [对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

```  
    1
   / \
  2   2
 / \ / \
3  4 4  3
```  

***示例***

**输入**: root = [1,2,2,3,4,4,3]

**输出**: true

```  
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root == NULL) return true;
        return isSymmetricalCor(root -> left, root -> right);
    }

private:
    bool isSymmetricalCor(TreeNode *L, TreeNode *R){
        if(L == NULL && R == NULL) return true;
        if(L == NULL || R == NULL) return false;
        if(L -> val != R -> val) return false;
        return isSymmetricalCor(L -> left, R -> right) && isSymmetricalCor(L -> right, R -> left);
    }
};
```  
```  
class Solution:
    def isSymmetricalCor(self, L: TreeNode, R: TreeNode) -> bool:
        if not L and not R:
            return true
        if not L or not R:
            return false
        if L.val != R.val:
            return self.isSymmetricalCor(L.left, R.right) and self.isSymmetricalCor(L.right, R.left)

    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return true
        return self.isSymmetricalCor(root.left, root.right)
```  


## 68-I. [二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

***示例***

![avatar](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png)

**输入**: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8

**输出**: 6

**解释**: 节点 2 和节点 8 的最近公共祖先是 6。

```  
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        while (true) {
            if (root->val > p->val && root->val > q->val)
                root = root->left;
            else if (root->val < p->val && root->val < q->val)
                root = root->right;
            else 
                return root;
        }
    }
};
```  
```  
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while (True):
            if root.val < p.val and root.val < q.val:
                root = root.right
            elif root.val > p.val and root.val > q.val:
                root = root.left
            else:
                return root
```  

## 68-II. [二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

***示例***

![avatar](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/15/binarytree.png)

**输入**: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1

**输出**: 3

**解释**: 节点 5 和节点 1 的最近公共祖先是 3。

```  
class Solution {
public:
    unordered_map<TreeNode*, TreeNode*> parent_map;
    unordered_set<TreeNode*> parents; //散列表，判断是否出现过
    void dfs(TreeNode *root) { //通过递归访问左右子树
        if (root->left != NULL) {
            parent_map[root->left] = root;
            dfs(root->left);
        }
        if (root->right != NULL) {
            parent_map[root->right] = root;
            dfs(root->right);
        }
    }
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        parent_map[root] = NULL; //建立父节点映射
        dfs(root);
        while (p != NULL) { //p的祖先都加入散列表
            parents.insert(p);
            p = parent_map[p];
        }
        while (q != NULL) { //q的祖先，一旦出现就是公共父节点
            if (parents.find(q) != parents.end()) //这样写就是若存在的意思
                return q;
            q = parent_map[q];
        }
        return NULL;
    }
};
```  
## 26. [树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)，B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如：给定的树 A:
```  
     3
    / \
   4   5
  / \
 1   2
```  
给定的树 A:
```  
   4 
  /
 1
```  
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

``` 
class Solution {
public:
    bool isSubStructure(TreeNode* A, TreeNode* B) {
        return (A != NULL && B != NULL) && (recur(A,B)||isSubStructure(A->left,B)||isSubStructure(A->right,B));
    }
    bool recur(TreeNode* A, TreeNode* B) {
        if (B == NULL) return true;
        if (A == NULL || A->val != B->val) return false;
        return recur(A->left, B->left) && recur(A->right, B->right);
    }
};
``` 

## 27. [二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)
请完成一个函数，输入一个二叉树，该函数输出它的镜像。

***示例***

**输入**: root = [4,2,7,1,3,6,9]

**输出**: [4,7,2,9,6,3,1]

递归交换左右子树

```  
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if(root == NULL) return NULL;
        TreeNode *temp = root->left;
        root->left = mirrorTree(root->right);
        root->right = mirrorTree(temp);
        return root;
    }
};
```  
```  
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        if not root:
            return root
        temp = root.left
        root.left = self.mirrorTree(root.right)
        root.right = self.mirrorTree(temp)
        return root
```  

## 32-I. [从上到下打印二叉树I](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)
从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

***示例***

**输入**: root = [3,9,20,null,null,15,7],

**输出**: [3,9,20,15,7]

```  
class Solution {
public:
    vector<int> levelOrder(TreeNode* root) {
        vector<int> res;
        if (root == NULL) return res;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            TreeNode *node = q.front();
            q.pop();
            res.push_back(node->val);
            if (node->left)
                q.push(node->left);
            if (node->right)
                q.push(node->right);
        }
        return res;
    }
};
```  
```  
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            res.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return res
```  
## 32-II. [从上到下打印二叉树II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，**每一层打印到一行**。

***示例***

**输入**: root = [3,9,20,null,null,15,7],

**输出**: 

[

  [3],
  
  [9,20],
  
  [15,7]
  
]

```  
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q;
        vector<vector<int> > res; 
        if (root == NULL) return res;
        q.push(root);
        while (!q.empty()) {
            vector<int> tmp;
            for (int i = q.size(); i > 0; i--) {
                TreeNode* node = q.front();
                q.pop();
                tmp.push_back(node->val);
                if (node->left != NULL) q.push(node->left);
                if (node->right != NULL) q.push(node->right);
            }
            res.push_back(tmp);
        }
        return res;
    }
};
```  
```  
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root: return []
        res, queue = [], collections.deque()
        queue.append(root)
        while queue:
            tmp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                tmp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(tmp)
        return res
```  
## 34. [二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)
输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的**根节点**开始往下一直到**叶节点**所经过的节点形成一条路径。

***示例***

**输入**: 给定如下二叉树，以及目标和 target = 22，
```
         5

        / \

       4   8
            
      /   / \

     11  13  4

    /  \    / \

   7    2  5   1
```

**输出**: 

[

   [5,4,11,2],
   
   [5,8,4,5]
   
]

**先序遍历**

```  
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        recur(root, target);
        return res;
    }

    void recur(TreeNode* root, int target1) {
        if (root == NULL) return;
        path.push_back(root->val);
        target1 -= root->val;
        if (target1 == 0 && root->left == NULL && root->right == NULL)
            res.push_back(path);
        recur(root->left, target1);
        recur(root->right, target1);
        path.pop_back();
    }
};
```  
```  
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        res, path = [], [] #path保存所有可能的路径
        def recur(root, target1):
            if not root: return
            path.append(root.val)
            target1 -= root.val
            if target1 == 0 and not root.left and not root.right:
                res.append(list(path))
            recur(root.left, target1)
            recur(root.right, target1)
            path.pop()
        
        recur(root, target)
        return res
```  

## 36. [二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：

![avatar](https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png)

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

![avatar](https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png)

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

``` 
class Solution {
public:
    Node* treeToDoublyList(Node* root) {
        if(root == nullptr) return nullptr;
        dfs(root);
        head->left = pre;
        pre->right = head;
        return head;
    }
private:
    Node *pre, *head;
    void dfs(Node* cur) {
        if(cur == nullptr) return;
        dfs(cur->left);
        if(pre != nullptr) pre->right = cur;
        else head = cur;
        cur->left = pre;
        pre = cur;
        dfs(cur->right);
    }
};
``` 

## 55-I. [二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，
```
    3
   / \
  9  20
    /  \
   15   7
```
返回它的最大深度 3 。

```  
class Solution {
public:
    int maxDepth(TreeNode* root) {
        list<TreeNode> queue;
        list<TreeNode> temp;
        if (root == NULL) return 0;
        int deep = 0;
        queue.push_back(*root);
        while (!queue.empty()){
            for (TreeNode node: queue){
                if (node.left != NULL) temp.push_back(*(node.left));
                if (node.right != NULL) temp.push_back(*(node.right));
            }
            queue = temp;
            temp.clear();
            deep ++;
        }
        return deep;
    }
};
```  
```  
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue, temp = [], []
        queue.append(root)
        deep = 0
        while queue:
            for node in queue:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            queue = temp
            temp = []
            deep += 1
        return deep
```  

## 55-II. [平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

***示例***

给定如下二叉树[3,9,20,null,null,15,7]

```  
    3
   / \
  9  20
    /  \
   15   7
```  
返回true

```  
class Solution {
public:
    int height(TreeNode* root) {
        if (root == NULL) return 0;
        else
            return max(height(root->left), height(root->right)) + 1;
    }
    bool isBalanced(TreeNode* root) {
        if (root == NULL) return true;
        else
            return abs(height(root->left)-height(root->right))<=1 && isBalanced(root->left) && isBalanced(root->right);
    }
};
```  
```  
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def height(root: TreeNode) -> int:
            if not root:
                return 0
            return max(height(root.left), height(root.right)) + 1
        if not root:
            return True
        else:
            return abs(height(root.left)-height(root.right))<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)
```  

## 54. [二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

给定一棵二叉搜索树，请找出其中第k大的节点。

***示例***

**输入**: root =  [3,1,4,null,2], k = 1

```  
   3
  / \
 1   4
  \
   2
```  
**输出**: 4

```  
反向深度优先遍历（反向中序遍历（右中左））

class Solution {
public:
    int ans = 0, count = 0;
    int kthLargest(TreeNode* root, int k) {
        reverse_dfs(root, k);
        return ans;
    }

    void reverse_dfs(TreeNode* root, int k){
        if (root == NULL) return;
        reverse_dfs(root->right, k);
        if (count == k) return;
        count ++;
        if (count == k) ans = root->val;
        reverse_dfs(root->left, k);
    }
};
```  
```  
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def dfs(root):
            if not root: return
            dfs(root.right)
            if self.k == 0: return
            self.k -= 1
            if self.k == 0: self.res = root.val
            dfs(root.left)

        self.k = k
        dfs(root)
        return self.res
```  
# 栈和队列

## 9. [用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

***示例***

**输入**: ["CQueue","appendTail","deleteHead","deleteHead"]，[[],[3],[],[]]

**输出**: [null,null,3,-1]
``` 
class CQueue {
public:

    stack<int> s1;
    stack<int> s2;
    CQueue() {

    }
    
    void appendTail(int value) {
        s1.push(value);
    }
    
    int deleteHead() {
        if(!s2.empty()){
            int temp = s2.top();
            s2.pop();
            return temp;
        } 
        if(s1.empty()) return -1;
        while(!s1.empty()){
            int temp = s1.top();
            s1.pop();   
            s2.push(temp);  
        }
    int temp = s2.top();
    s2.pop();
    return temp;
    }
};
``` 
``` 
class CQueue(object):

    def __init__(self):
        self.s1, self.s2 = [],[]

    def appendTail(self, value):
        """
        :type value: int
        :rtype: None
        """
        self.s1.append(value)

    def deleteHead(self):
        """
        :rtype: int
        """
        if self.s2:
            return self.s2.pop()
        if not self.s1:
            return -1

        while self.s1:
            self.s2.append(self.s1.pop())
        return self.s2.pop()
  ``` 

## 30. [包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

***示例***
  ``` 
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
  ``` 
 ``` 
class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {

    }
    
    stack<int> S1, minS2;

    void push(int x) {
        S1.push(x);
        if(minS2.empty()) minS2.push(x);
        else if(minS2.top() > x && !minS2.empty()) minS2.push(x);
        else minS2.push(minS2.top());
    }
    
    void pop() {
        S1.pop();
        minS2.pop();
    }
    
    int top() {
        return S1.top();
    }
    
    int min() {
        return minS2.top();
    }
};
 ``` 
  ``` 
 class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.A, self.B = [],[]

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        self.A.append(x)
        if not self.B or self.B[-1] > x:
            self.B.append(x)
        elif self.B[-1] <= x:
            self.B.append(self.B[-1])

    def pop(self):
        """
        :rtype: None
        """
        self.B.pop()
        self.A.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.A[-1]

    def min(self):
        """
        :rtype: int
        """
        return self.B[-1]
  ``` 

## 31. [栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

***示例***

**输入**：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]

**输出**：true

**解释**：我们可以按以下顺序执行：

push(1), push(2), push(3), push(4), pop() -> 4, push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1

  ``` 
class Solution {
public:
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
        stack<int> s;
        int i=0, j;
        for(j = 0; j < pushed.size(); j++)
        {
            s.push(pushed[j]);
            while(!s.empty() && s.top() == popped[i]){
                s.pop();
                i++;
            }
        }
        return s.empty();
    }
};
 ```  
 ``` 
 class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        i = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
        return not stack 
 ``` 
## 33. [二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

参考以下这棵二叉搜索树：

 ``` 
     5
    / \
   2   6
  / \
 1   3
  ``` 
  
**输入**: [1,6,3,2,5]

**输出**: false

  ``` 
class Solution {
public:
    bool verifyPostorder(vector<int>& postorder) {
        stack<int> s;
        int root = 10000000;
        for (int i = postorder.size() - 1; i >= 0; i--) {
            if (postorder[i] > root) return false;
            while (!s.empty() && s.top() > postorder[i]) {
                root = s.top();
                s.pop();
            }
            s.push(postorder[i]);
        }
        return true;
    }
};
``` 

``` 
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        stack, root = [], float("+inf")
        for i in range(len(postorder) - 1, -1, -1):
            if postorder[i] > root: return False
            while(stack and postorder[i] < stack[-1]):
                root = stack.pop()
            stack.append(postorder[i])
        return True
``` 

# 数组与字符串

## 62. [圆圈中最后剩下的数字（约瑟夫环）](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

***示例***

**输入**：n = 5, m = 3

**输出**：3

``` 
class Solution {
public:
    int lastRemaining(int n, int m) {
        int f = 0;
        for (int i = 2; i != n + 1; i++)
            f = (m + f) % i;
        return f;
    }
};
``` 

``` 
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        f = 0
        for i in range(2, n+1):
            f = (m + f) % i;
        return f
``` 

## 17. [打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

***示例***

**输入**：n = 1

**输出**：[1,2,3,4,5,6,7,8,9]

``` 
class Solution {
public:
    vector<int> printNumbers(int n) {
        vector<int> ans;
        for (int i = 1; i < pow(10, n); i++)
            ans.push_back(i);
        return ans;
    }
};
``` 
``` 
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        ans = []
        for i in range(1, 10**n):
            ans.append(i);
        return ans
``` 

## 39. [数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/submissions/)

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

***示例***

**输入**：[1, 2, 3, 2, 2, 2, 5, 4, 2]

**输出**：2

``` 
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int x = 0, votes = 0;
        for(int num: nums){
            if(votes == 0) x = num;
            votes += num == x ? 1 : -1;
        }
        return x;
    }
};
``` 
``` 
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0:
                x = num
            if num == x:
                votes += 1
            else:
                votes += -1
        return x
``` 
## 5. [替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

***示例***

**输入**：s = "We are happy."

**输出**："We%20are%20happy."

``` 
class Solution {
public:
    string replaceSpace(string s) {
        int count = 0, len = s.size();
        for (char c : s) 
            if (c == ' ') count++; // 统计空格数量
        s.resize(len + 2 * count);  // 修改 s 长度
        // 倒序遍历修改
        for(int i = len - 1, j = s.size() - 1; i < j; i--, j--) {
            if (s[i] != ' ')
                s[j] = s[i];
            else {
                s[j - 2] = '%';
                s[j - 1] = '2';
                s[j] = '0';
                j -= 2;
            }
        }
        return s;
    }
};
``` 
``` 
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == ' ': 
                res.append("%20")
            else: 
                res.append(c)
        return "" .join(res)
``` 

## 21. [调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。  

***示例***

**输入**：nums = [1,2,3,4]

**输出**：[1,3,2,4] 

``` 
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int begin = 0, end = nums.size() - 1;
        while(begin < end){
            if(nums[begin] % 2 == 1)
                begin++;
            else if(nums[end] % 2 == 0)
                end--;
            if(begin < end){
                int temp = nums[begin];
                nums[begin] = nums[end];
                nums[end] = temp;
            }
        }
    return nums;
    }
};
``` 
``` 
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        begin = 0
        end = len(nums) - 1
        while(begin < end):
            if nums[begin] % 2 == 1:
                begin = begin + 1
            elif nums[end] % 2 == 0:
                end = end - 1
            if begin < end:
                temp = nums[begin]
                nums[begin] = nums[end]
                nums[end] = temp
        return nums
``` 

# 查找
## 4. [二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)

在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

***示例***

现有矩阵 matrix 如下：

``` 
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
``` 

给定 target = 5，返回 true。给定 target = 20，返回 false。

从左下角或者右上角开始查找

``` 
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i = matrix.size() - 1, j = 0;
        while(i >= 0 && j < matrix[0].size())
        {
            if(matrix[i][j] > target)
                i--;
            else if(matrix[i][j] < target)
                j++;
            else return true;
        }
        return false;
    }
};
``` 
``` 
class Solution(object):
    def findNumberIn2DArray(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if matrix == []:
            return False
        i = len(matrix) - 1
        j = 0
        while(i >=0 and j < len(matrix[0])):
            if matrix[i][j] > target:
                i = i - 1
            elif matrix[i][j] < target:
                j = j + 1
            else:
                return True
        return False
``` 

## 11. [旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

***示例***

**输入**：[3,4,5,1,2]

**输出**：1

类似于二分查找

``` 
class Solution {
public:
    int minArray(vector<int>& numbers) {
        int low = 0;
        int high = numbers.size() - 1;
        while (low < high) {
            int pivot = low + (high - low) / 2;
            if (numbers[pivot] < numbers[high]) {
                high = pivot;
            }
            else if (numbers[pivot] > numbers[high]) {
                low = pivot + 1;
            }
            else {
                high -= 1;
            }
        }
        return numbers[low];
    }
};
``` 

``` 
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        low, high = 0, len(numbers) - 1
        while low < high:
            pivot = low + (high - low) // 2
            if numbers[pivot] < numbers[high]:
                high = pivot 
            elif numbers[pivot] > numbers[high]:
                low = pivot + 1
            else:
                high -= 1
        return numbers[low]
``` 

# 位运算
## 53-II. [0~n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

***示例***

**输入**：[0,1,3]

**输出**：2

位运算：相同两个数异或为0，所有数与0异或结果不变，所以将数组里面所有的数异或，然后再将结果异或0~n-1之间的所有的数，最后只有一个数只出现了一次，就是最后的结果。
``` 
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int ans = 0;
        for(int i = 0; i < nums.size(); i ++)
            ans ^= nums[i];
        for(int i = 0; i <= nums.size(); i ++)
            ans ^= i;
        return ans;
    }   
};
``` 
二分查找
``` 
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int i = 0, j = nums.size() - 1;
        while(i <= j){
            int m = (i + j) / 2;
            if(nums[m] == m) i = m + 1;
            else j = m -1;
        }
        return i;
    }
};
``` 

## 15.[三数之和](https://leetcode-cn.com/problems/3sum/)

给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

***示例***

**输入**：nums = [-1,0,1,2,-1,-4]

**输出**：[[-1,-1,2],[-1,0,1]]

``` 
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        vector<vector<int>> ans;
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    ans.push_back({nums[first], nums[second], nums[third]});
                }
            }
        }
        return ans;
    }
};
``` 


## 56-I. [数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

***示例***

**输入**：nums = [4,1,4,6]

**输出**：[1,6] 或 [6,1]

``` 
class Solution {
public:
    vector<int> singleNumbers(vector<int>& nums) {
        int x = 0, y = 0, n = 0, m = 1;
        for(int num : nums)         // 1. 遍历异或
            n ^= num;
        while((n & m) == 0)         // 2. 循环左移，计算 m
            m <<= 1;
        for(int num : nums) {       // 3. 遍历 nums 分组
            if(num & m) x ^= num;   // 4. 当 num & m != 0
            else y ^= num;          // 4. 当 num & m == 0
        }
        return vector<int> {x, y};  // 5. 返回出现一次的数字
    }
};

``` 

## 56-II. [数组中数字出现的次数II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

***示例***

**输入**：nums = [3,4,3,3]

**输出**：4

因为数组中只有一个数出现了一次，那么各个二进制位为1的个数 % 3 便能求出这个数哪些位置为1， 最后再将其转换为十进制

```
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int bits[32] = {0};
        for(int i = 0; i < nums.size(); i++){
            int j = 0;
            //得到各个二进制位为1的有多少个
            while(nums[i]){
                bits[j] += nums[i] % 2;
                nums[i] /= 2;
                j++;
            }
        }
        int ans = 0;
        for(int i = 0; i < 32; i++){
            //利用%3 来求得对应位置上有没有1 有的话乘对应的 2 的i次方
            ans += (1 << i) *(bits[i] % 3);
        }
        return ans;
    }
};

```

# 深度优先搜索, 广度优先搜索
## 200. [岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

***示例***

**输入**：
``` 
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
``` 

**输出**：1

BFS，使用队列
``` 
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int nr = grid.size();
        if (!nr) return 0;
        int nc = grid[0].size();

        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    grid[r][c] = '0';
                    queue<pair<int, int>> neighbors;
                    neighbors.push({r, c});
                    while (!neighbors.empty()) {
                        auto rc = neighbors.front();
                        neighbors.pop();
                        int row = rc.first, col = rc.second;
                        if (row - 1 >= 0 && grid[row-1][col] == '1') {
                            neighbors.push({row-1, col});
                            grid[row-1][col] = '0';
                        }
                        if (row + 1 < nr && grid[row+1][col] == '1') {
                            neighbors.push({row+1, col});
                            grid[row+1][col] = '0';
                        }
                        if (col - 1 >= 0 && grid[row][col-1] == '1') {
                            neighbors.push({row, col-1});
                            grid[row][col-1] = '0';
                        }
                        if (col + 1 < nc && grid[row][col+1] == '1') {
                            neighbors.push({row, col+1});
                            grid[row][col+1] = '0';
                        }
                    }
                }
            }
        }

        return num_islands;
    }
};
``` 
## 13. [机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

***示例***

**输入**：m = 2, n = 3, k = 1

**输出**：3

BFS
``` 
class Solution {
    // 计算 x 的数位之和
    int get(int x) {
        int res=0;
        for (; x; x /= 10) {
            res += x % 10;
        }
        return res;
    }
public:
    int movingCount(int m, int n, int k) {
        if (!k) return 1;
        queue<pair<int,int> > Q;
        // 向右和向下的方向数组
        int dx[2] = {0, 1};
        int dy[2] = {1, 0};
        vector<vector<int> > vis(m, vector<int>(n, 0));
        Q.push(make_pair(0, 0));
        vis[0][0] = 1;
        int ans = 1;
        while (!Q.empty()) {
            auto [x, y] = Q.front();
            Q.pop();
            for (int i = 0; i < 2; ++i) {
                int tx = dx[i] + x;
                int ty = dy[i] + y;
                if (tx < 0 || tx >= m || ty < 0 || ty >= n || vis[tx][ty] || get(tx) + get(ty) > k) continue;
                Q.push(make_pair(tx, ty));
                vis[tx][ty] = 1;
                ans++;
            }
        }
        return ans;
    }
};
``` 
