
interface Marks {
    void marksinMaths(int pass);

    void marksinJava(int pass);

    void marksinEnglish(int fail);
}
class Marksheet implements Marks {
    int marksinMath = 80;
    int marksinJava = 88;
    int marksinEnglish = 20;

    public void marksinMaths(int pass) {
        System.out.println("You passed in Maths with: " + pass);
    }

    public void marksinJava(int pass) {
        System.out.println("You passed in Java with: " + pass);
    }

    public void marksinEnglish(int fail) {
        System.out.println("You failed in English with: " + fail);
    }
}
// Abstract class extending the concrete class/// humne bss yaha marksheet_2 kr
// diya and extends
// MarkSHEET KR DIYA END ME ABSTRACT CLASS BNAYA JATA HAI IN SB ME INHERITANCE
// ME

abstract class Marksheet_2 extends Marksheet {
    public abstract void marksinMaths(int pass);

    public abstract void marksinJava(int pass);

    public abstract void marksinEnglish(int pass);
}
/// extra in heritance 
abstract class MobileData extends Marksheet_2 {

    public abstract void MobileonCharge(int on);

    public abstract void Mobileonnotconnected(int off);

}




// Main class to run the program // kr skte hai AND BSS YAHI OBJECT BNANE WALA
// KAAM
public class inheritance_in_interface {
    public static void main(String[] args) {
        Marksheet ek = new Marksheet();
        ek.marksinMaths(100);
        ek.marksinJava(80);
        ek.marksinEnglish(20);

    }
}
abstract class Traintiket extends MobileData{
    public abstract void LateTrain(int miss);
    public abstract void EarlyTrain(int onTime);


}
abstract class BussTicket extends Trainticket {
    public abstract void BussLate(int late);
    public abstract void EarlyBuss(int early);
    

}







