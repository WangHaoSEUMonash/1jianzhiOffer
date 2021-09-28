# 链表
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
