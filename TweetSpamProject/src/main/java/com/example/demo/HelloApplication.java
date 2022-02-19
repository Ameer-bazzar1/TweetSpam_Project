package com.example.demo;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.input.KeyCombination;
import javafx.stage.Stage;
import java.io.IOException;
public class HelloApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("View.fxml"));
        Scene scene = new Scene(fxmlLoader.load());
        stage.setTitle("Welcome");
        stage.setScene(scene);
        stage.setFullScreenExitKeyCombination(KeyCombination.NO_MATCH);
        stage.setResizable(false);
        stage.setScene(scene);
        stage.setMaximized(false);
        stage.show();
    }

    public static void main(String[] args) {
        launch();
    }
}