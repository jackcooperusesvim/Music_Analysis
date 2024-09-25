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

	df := SerArrToDf(sers)
	if df.err != nil {
		t.Errorf("Error creating DataFrame: %s", df.err)
	}
	if df.shape.cols != 3 {
		t.Errorf("DataFrame has %d columns, not 3 {%+v}", df.shape.cols, df)
	}
	if df.shape.rows != 3 {
		t.Errorf("DataFrame has %d rows, not 3 {%+v}", df.shape.rows, df)
	}
	for i, row := range df.data {
		if len(row) != len(expected_df.data[i]) {
			t.Errorf("DataFrame is not equal to expected DataFrame exp:{%+v}\n\nact:{%+v}", expected_df, df)
		}
		for j, v := range row {
			if v != expected_df.data[i][j] {
				t.Errorf("DataFrame is not equal to expected DataFrame exp:{%+v}\n\nact:{%+v}", expected_df, df)
			}
		}
	}
}

func TestCalcShape(t *testing.T) {
	sers := []Series{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
		[]float64{7, 8, 9},
	}
	df := SerArrToDf(sers)
	if df.err != nil {
		t.Errorf("Error creating DataFrame: %s", df.err)
	}
	df.CalcShape()
	if df.err != nil {
		t.Errorf("Error calculating shape: %s", df.err)
	}
	if df.shape.cols != 3 {
		t.Errorf("DataFrame has %d columns, not 3", df.shape.cols)
	}
	if df.shape.rows != 3 {
		t.Errorf("DataFrame has %d rows, not 3", df.shape.rows)
	}
}
