import java.util.*;

class Solution {

    public ArrayList<String> findPermutation(String S) {
        ArrayList<String> ans = new ArrayList<>();
        char[] arr = S.toCharArray();
        
        Arrays.sort(arr);
        
        boolean[] visited = new boolean[arr.length];
        
        helper(arr, "", visited, ans);
        return ans;
    }

    private void helper(char[] arr, String curr, boolean[] visited, ArrayList<String> ans) {
        if(curr.length() == arr.length) {
            ans.add(curr);
            return;
        }

        for(int i = 0; i < arr.length; i++) {

            if(visited[i]) continue;

        
            if(i > 0 && arr[i] == arr[i-1] && !visited[i-1]) continue;

            visited[i] = true;
            helper(arr, curr + arr[i], visited, ans);
            visited[i] = false;
        }
    }
}
