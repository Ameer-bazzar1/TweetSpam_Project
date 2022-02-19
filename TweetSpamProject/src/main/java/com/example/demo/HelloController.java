package com.example.demo;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import java.io.File;
import java.io.IOException;
public class HelloController {
    @FXML
    Button Browse;
    @FXML
    TextField BrowseTextField;
    @FXML
    TextField Sensitivity;
    @FXML
    TextField precision;
    @FXML
    TextField f1;
    @FXML
    TextField accuracy;
    @FXML
    TextField ROC;
    @FXML
    RadioButton Tree;
    @FXML
    RadioButton Multi;
    @FXML
    RadioButton Naive;
    @FXML
    protected void onBrowseButtonClick() throws IOException {
        Stage stage=new Stage();

        Browse.setOnAction(actionEvent -> {

            FileChooser fileChooserShares = new FileChooser();
            fileChooserShares.setTitle("Select daily price file .csv");
            fileChooserShares.getExtensionFilters().addAll(
                    new FileChooser.ExtensionFilter("Csv Files", "*.csv"),
                    new FileChooser.ExtensionFilter("Csv files (' csv')", "*.csv")
            );
            File selectedFile = fileChooserShares.showOpenDialog(stage);
            if (String.valueOf(selectedFile).equals("null")) {
                return;
            }
            else{
                BrowseTextField.setText(selectedFile.toString());
            }
        });
    }
    @FXML
    protected void onReadButtonClick() throws IOException, InterruptedException {
        Alert alert=new Alert(Alert.AlertType.ERROR);
        if (BrowseTextField.getText()==null ||BrowseTextField.getText().isBlank() || BrowseTextField.getText().isBlank() ){
            alert.setContentText("Path can't be Empty.");
            alert.show();
            return;
        }else {
            Alert alert3=new Alert(Alert.AlertType.ERROR);
            if (Tree.isSelected()){
                SomeJavaProgram.start("Tree",BrowseTextField.getText());
                Sensitivity.setText(SomeJavaProgram.Sensitivity);
                precision.setText(SomeJavaProgram.precision);
                f1.setText(SomeJavaProgram.f1);
                accuracy.setText(SomeJavaProgram.accuracy);
                ROC.setText(SomeJavaProgram.ROC);
            }else if (Naive.isSelected()){
                SomeJavaProgram.start("Naive",BrowseTextField.getText());
                Sensitivity.setText(SomeJavaProgram.Sensitivity);
                precision.setText(SomeJavaProgram.precision);
                f1.setText(SomeJavaProgram.f1);
                accuracy.setText(SomeJavaProgram.accuracy);
                ROC.setText(SomeJavaProgram.ROC);
            }else if (Multi.isSelected()){
                SomeJavaProgram.start("Multi",BrowseTextField.getText());
                Sensitivity.setText(SomeJavaProgram.Sensitivity);
                precision.setText(SomeJavaProgram.precision);
                f1.setText(SomeJavaProgram.f1);
                accuracy.setText(SomeJavaProgram.accuracy);
                ROC.setText(SomeJavaProgram.ROC);
            }else {
                alert3.setContentText("Model can't be Empty.");
                alert3.show();
                return;
            }
        }
    }
}