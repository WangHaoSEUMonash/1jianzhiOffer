# 链表
## 18. 删除链表的节点
给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。返回删除后的链表的头节点
*示例*
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
*示例*
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
