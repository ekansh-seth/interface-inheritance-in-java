import java.util.Scanner;

public class array_6 {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);  // 
        System.out.println("Enter 8 elements of the array:");

        int[] arr = new int[8];  

        //  Input
        for (int i = 0; i < 8; i++) {
            arr[i] = sc.nextInt();
        }

        //  Sum 
        int sum = 0;
        for (int i = 0; i < 8; i++) {
            sum += arr[i];
        }

        System.out.println(sum);

        sc.close(); // optional hai but esse code aur aacha lgta hai 
    }
}

    

