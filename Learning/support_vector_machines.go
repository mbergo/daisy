package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/ewalker544/libsvm-go"
	"gonum.org/v1/gonum/stat"
)

type iris struct {
	sepalLength float64
	sepalWidth  float64
	petalLength float64
	petalWidth  float64
	class       string
}

func main() {
	irisData, err := loadIrisData("iris.csv")
	if err != nil {
		log.Fatalf("Error loading Iris dataset: %v", err)
	}

	problem, labels := prepareSVMData(irisData)

	param := libsvm.NewParameter()
	param.SvmType = libsvm.C_SVC
	param.KernelType = libsvm.RBF
	param.C = 1
	param.Gamma = 1

	model, err := libsvm.NewModel(param)
	if err != nil {
		log.Fatalf("Error creating SVM model: %v", err)
	}
	model.Train(problem)

	predictions := model.Predict(problem)
	accuracy := calculateAccuracy(labels, predictions)

	fmt.Printf("SVM Training Accuracy: %.2f%%\n", accuracy*100)
}

func loadIrisData(filename string) ([]iris, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var irisData []iris
	for _, record := range records[1:] {
		sepalLength, _ := strconv.ParseFloat(record[0], 64)
		sepalWidth, _ := strconv.ParseFloat(record[1], 64)
		petalLength, _ := strconv.ParseFloat(record[2], 64)
		petalWidth, _ := strconv.ParseFloat(record[3], 64)
		class := record[4]

		irisData = append(irisData, iris{sepalLength, sepalWidth, petalLength, petalWidth, class})
	}

	return irisData, nil
}

func prepareSVMData(irisData []iris) (problem libsvm.Problem, labels []float64) {
	problem.Elements = make([][]libsvm.SvmNode, len(irisData))
	labels = make([]float64, len(irisData))

	for i, row := range irisData {
		problem.Elements[i] = []libsvm.SvmNode{
			{Index: 1, Value: row.sepalLength},
			{Index: 2, Value: row.sepalWidth},
			{Index: 3, Value: row.petalLength},
			{Index: 4, Value: row.petalWidth},
		}

		switch row.class {
		case "Iris-setosa":
			labels[i] = 0
		case "Iris-versicolor":
			labels[i] = 1
		case "Iris-virginica":
			labels[i] = 2
		}
	}

	problem.Labels = labels

	return problem, labels
}

func calculateAccuracy(yTrue, yPred []float64) float64 {
	correct := 0
	for i := 0
	if yTrue[i] == yPred[i] {
		correct++
	}
}

accuracy := float64(correct) / float64(len(yTrue))
return accuracy
}
