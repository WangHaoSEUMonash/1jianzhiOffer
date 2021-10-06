# 基本排序

## 912. [排序数组](https://leetcode-cn.com/problems/sort-an-array/)
给你一个整数数组 nums，请你将该数组升序排列。

***示例***

**输入**: nums = [5,2,3,1]

**输出**: [1,2,3,5]

### 快速排序 ***O(nlogn)，不稳定***

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

# 链表

## 6. [从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

***示例***

**输入**: head = [1,3,2]

**输出**: [2,3,1]

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


