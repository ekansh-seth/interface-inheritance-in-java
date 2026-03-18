class Solution {
    public Node addTwoLists(Node head1, Node head2) {
        // code here
        Node temp1 = reverseLinkedList(head1);
         Node temp2 = reverseLinkedList(head2);
         int carry = 0;
         int sum = 0;
         Node ansHead = null;
         
         while(temp1 != null || temp2 != null || carry != 0) {
             sum = carry;
             
             if(temp1 != null) {
                 sum = sum + temp1.data;
                 temp1 = temp1.next;
             }
             
             if(temp2 != null) {
                 sum = sum + temp2.data;
                 temp2 = temp2.next;
             }
             
             carry = sum / 10;
             Node nextNode = new Node(sum % 10);
             nextNode.next = ansHead;
             ansHead = nextNode;
         }
         
        while(ansHead != null && ansHead.data == 0 && ansHead.next != null) {
            ansHead = ansHead.next;
        }
        
         return ansHead;
    }
    
    public Node reverseLinkedList(Node head) {
        Node prev = null;
        Node next = null;
        Node curr = head;
        
        while(curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        
        return prev;
    }
}