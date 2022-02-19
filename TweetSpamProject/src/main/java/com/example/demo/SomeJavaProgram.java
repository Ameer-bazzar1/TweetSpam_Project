package com.example.demo;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;

import java.io.*;

public class SomeJavaProgram {

   static String ROC;

   static String Sensitivity;

   static String precision;

  static   String f1;

   static String accuracy;

    public static void start(String model,String path) throws IOException, InterruptedException {
        ProcessBuilder process;
        if (model.equals("Tree")){
            process = new ProcessBuilder("C:\\Users\\Ameer\\AppData\\Local\\Programs\\Python\\Python39\\python.exe","src/main/Python/main.py","Tree",path);
        }else if (model.equals("Naive")){
            process = new ProcessBuilder("C:\\Users\\Ameer\\AppData\\Local\\Programs\\Python\\Python39\\python.exe","src/main/Python/main.py","Naive",path);
        }else {
            process = new ProcessBuilder("C:\\Users\\Ameer\\AppData\\Local\\Programs\\Python\\Python39\\python.exe","src/main/Python/main.py","Multi",path);
        }
        Process p=process.start();
        p.waitFor();
        BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
        BufferedReader reader1 = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        String line="";
        while ((line = reader1.readLine()) != null){
            System.out.println(line);
        }
        Sensitivity=  reader.readLine();
        precision=  reader.readLine();
        f1= reader.readLine();
        accuracy= reader.readLine();
        ROC= reader.readLine();
    }
    }
