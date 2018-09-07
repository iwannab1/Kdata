package kdata.fp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class Null {

    public Integer toInt(String s){
        try{
            return(Integer.parseInt(s));
        }catch(NumberFormatException e){
            return(null);
        }
    }

    public static void main(String[] args){
        //System.out.println(Integer.parseInt("aaa"));
        Null n = new Null();
        String[] bagitem = new String[]{"1", "2", "foo", "3", "bar"};
        List<String> bag = Arrays.asList(bagitem);

        int sum = bag.stream().mapToInt(Integer::parseInt).sum();
        System.out.println(sum);
    }
}
