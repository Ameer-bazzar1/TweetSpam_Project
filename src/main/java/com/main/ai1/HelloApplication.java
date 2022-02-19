package com.main.ai1;
import java.io.*;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.TableView;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Scanner;
import java.util.regex.Pattern;

public class HelloApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("hello-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load());
        stage.setTitle("Hello!");
        stage.setScene(scene);
        stage.show();

    }

    public static void main(String[] args) throws IOException, InterruptedException {
        ProcessBuilder processBuilder=new ProcessBuilder("C:\\Users\\Ameer\\AppData\\Local\\Programs\\Python\\Python39\\python.exe","src/main/python/main.py");
        processBuilder.redirectErrorStream(true);
        Process process=processBuilder.start();
        System.out.println("Python script finished successfully");
        launch();
    }
}