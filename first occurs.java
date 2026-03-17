class Solution {
    ArrayList<Integer> find(int arr[], int x) {
        // code here
       int  n=arr.length;
        ArrayList<Integer> a=new ArrayList<Integer>();
        int i=-1;
        int res=0,j=0;
        for(j=0;j<n;j++){
            if(arr[j]==x){
                if(i==-1){
                    i=j;
                }
                res++;
                
            }
        }
        if(res==0){
            a.add(i);
            a.add(i);
        }
        else{
            a.add(i);
            a.add(i+res-1);
        }
        return a;
        
    }
}