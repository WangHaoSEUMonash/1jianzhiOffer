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
