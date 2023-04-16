package main

import (
	"fmt"
	"log"
	"strings"

	"github.com/james-bowman/tfidf"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	userQuestions := []string{
		"What is the weather like today?",
		"How do I cook pasta?",
		"What are the best vacation spots?",
		"How to create a neural network?",
		"What is the temperature outside?",
		"Best Italian recipes to try at home?",
		"Top holiday destinations this year?",
		"Implementing machine learning algorithms",
	}

	vectorizer := tfidf.NewVectorizer()
	questionsMatrix, err := vectorizer.FitTransform(strings.NewReader(strings.Join(userQuestions, "\n")))
	if err != nil {
		log.Fatalf("Error vectorizing user questions: %v", err)
	}

	k := 3
	centroids, clusters := kmeans(questionsMatrix, k)

	fmt.Printf("Centroids:\n%v\n", centroids)
	fmt.Printf("Clusters:\n%v\n", clusters)
}

func kmeans(questionsMatrix mat.Matrix, k int) (centroids *mat.Dense, clusters []int) {
	rows, _ := questionsMatrix.Dims()

	centroids = mat.NewDense(k, rows, nil)
	clusters = make([]int, rows)

	// Initialize centroids
	for i := 0; i < k; i++ {
		centroids.SetRow(i, mat.Row(nil, i, questionsMatrix))
	}

	changed := true
	for changed {
		changed = false

		// Assign points to centroids
		for i := 0; i < rows; i++ {
			minDist := mat.Norm(mat.NewVecDense(rows, nil), 2)
			minIndex := 0

			for j := 0; j < k; j++ {
				centroidRow := mat.Row(nil, j, centroids)
				pointRow := mat.Row(nil, i, questionsMatrix)
				distance := floats.Distance(centroidRow, pointRow, 2)

				if distance < minDist {
					minDist = distance
					minIndex = j
				}
			}

			if clusters[i] != minIndex {
				clusters[i] = minIndex
				changed = true
			}
		}

		// Update centroids
		for i := 0; i < k; i++ {
			sum := mat.NewVecDense(rows, nil)
			count := 0.0

			for j, cluster := range clusters {
				if cluster == i {
					sum.AddVec(sum, mat.NewVecDense(rows, mat.Row(nil, j, questionsMatrix)))
					count++
				}
			}

			if count != 0 {
				sum.ScaleVec(1/count, sum)
				centroids.SetRow(i, mat.Row(nil, 0, sum))
			}
		}
	}

	return
}
