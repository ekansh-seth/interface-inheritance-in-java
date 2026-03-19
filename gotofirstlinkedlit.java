class Solution {
    public Node moveToFront(Node head) {
        
        if(head == null || head.next == null) return head;

        Node prev = null;
        Node curr = head;

        // go to last node
        while(curr.next != null) {
            prev = curr;
            curr = curr.next;
        }

        // curr = last node
        // prev = second last node

        prev.next = null;
        curr.next = head;
        head = curr;

        return head;
    }
}