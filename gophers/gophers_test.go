package gophers

import (
	"testing"
)

func TestSerArrToDf(t *testing.T) {
	sers := []Series{

		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}
	expected_df := DataFrame{
		data: []Series{
			[]float64{1, 2, 3},
			[]float64{4, 5, 6},
			[]float64{7, 8, 9},
		},
		shape: Shape{
			cols: 3,
			rows: 3,
		},
	}

	df, err := SerArrToDf(sers)
	if err != nil {
		t.Errorf("Error creating DataFrame: %s", err)
	}
	if df.shape.cols != 3 {
		t.Errorf("DataFrame has %d columns, not 3", df.shape.cols)
	}
	if df.shape.rows != 3 {
		t.Errorf("DataFrame has %d rows, not 3", df.shape.rows)
	}
	for i, row := range df.data {
		if len(row) != len(expected_df.data[i]) {
			t.Errorf("DataFrame is not equal to expected DataFrame")
		}
		for j, v := range row {
			if v != expected_df.data[i][j] {
				t.Errorf("DataFrame is not equal to expected DataFrame")
			}
		}
	}
}

func TestSeriesDistance(t *testing.T) {
	s1 := []float64{1, 1, 5}
	s2 := []float64{1, 3, 3}
	expected := 16.0
	actual := SeriesDistance(s1, s2)
	if actual != expected {
		t.Errorf("Distance between %v and %v is %f, not %f", s1, s2, actual, expected)
	}
}

func TestCalcShape(t *testing.T) {
	sers := []Series{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}
	df, err := SerArrToDf(sers)
	if err != nil {
		t.Errorf("Error creating DataFrame: %s", err)
	}
	shape, err := df.CalcShape()
	if err != nil {
		t.Errorf("Error calculating shape: %s", err)
	}
	if shape.cols != 3 {
		t.Errorf("DataFrame has %d columns, not 3", shape.cols)
	}
	if shape.rows != 3 {
		t.Errorf("DataFrame has %d rows, not 3", shape.rows)
	}
}

func TestKnnDistance(t *testing.T) {
	knn := Build_Knn(2)

	x_df, err := SerArrToDf(
		[]Series{
			[]float64{1, 1, 1},
			[]float64{2, 2, 2},
			[]float64{3, 3, 3},
		})
	dist_df, err := SerArrToDf(
		[]Series{
			[]float64{1, 1, 1},
			[]float64{2, 2, 2},
			[]float64{3, 3, 3},
		})
	exp_dist_df, err := SerArrToDf(
		[]Series{
			[]float64{0, 3, 0},
			[]float64{0, 0, 0},
			[]float64{0, 0, 0},
		})
	if err != nil {
		t.Errorf("Error creating DataFrame: %s", err)
	}

	actual_dist_df, err := knn.Distance(x_df)
	if err != nil {
		t.Errorf("Error calculating distance: %s", err)
	}
	if exp_dist_df.shape != actual_dist_df.shape {
		t.Errorf("Distance DataFrame has %d columns, not %d", actual_dist_df.shape.cols, exp_dist_df.shape.cols)
	}
}
