package gophers

import (
	"errors"
	"fmt"
	"math"
)

type Series []float64
type DataFrame struct {
	data  []Series
	shape Shape
	err   error
}
type Shape struct {
	cols int
	rows int
}

type Knn struct {
	train_x   DataFrame
	train_y   DataFrame
	in_shape  Shape
	out_shape Shape
	k         int
}

type Model interface {
	fit(x DataFrame, y DataFrame) (Model, error)
	predict_regress(x DataFrame) (y DataFrame)
	predict_classify(x DataFrame) (y DataFrame)
}

func SerArrToDf(dta []Series) (DataFrame, error) {
	out := DataFrame{
		data: dta,
	}
	_, err := out.CalcShape()
	return out, err
}

func SeriesDistance(s1 Series, s2 Series) (out float64) {
	//This is squared euclidean distance
	for i, v := range s1 {
		out += (v - s2[i]) * (v - s2[i])
	}
	return out
}
func (df *DataFrame) CalcShape() (Shape, error) {

	shape := Shape{}
	if df.err != nil {
		return Shape{}, df.err
	}
	if len(df.data) == 0 {
		df.err = errors.New("DataFrame is empty, and therefore invalid")
		return Shape{}, df.err
	}

	shape.cols = len(df.data)

	sub_len := len(df.data[0])
	for i, v := range df.data {
		if len(v) != sub_len {
			df.err = errors.New(fmt.Sprintf("DataFrame has inconsistent shape: column 0->%d of length %d, but column %d is of length %d", i-1, sub_len, i, len(v)))
			return Shape{}, df.err
		}
	}

	shape.rows = sub_len
	return shape, nil
}
func (knn *Knn) Distance(x DataFrame) (DataFrame, error) {
	actual_input_shape, err := x.CalcShape()
	if err != nil {
		return DataFrame{}, err
	}
	if actual_input_shape != knn.in_shape {
		return DataFrame{}, errors.New(fmt.Sprintf("Shape on incoming DataFrame (%d,%d) does not match proper input shape (%d,%d)", actual_input_shape.rows, actual_input_shape.cols, knn.in_shape.cols, knn.in_shape.rows))
	}

	out_data := make([]Series, knn.in_shape.rows)

	for i, series_train := range knn.train_x.data {
		pred_distances := make(Series, knn.in_shape.rows)
		for j, series_pred := range x.data {
			pred_distances[j] = SeriesDistance(series_train, series_pred)
		}
		out_data[i] = pred_distances
	}

	out, err := SerArrToDf(out_data)

	if err != nil {
		return DataFrame{}, err
	}
	return out, nil
}

func Build_Knn(k int) Knn {
	return Knn{k: k}
}

// func (knn *Knn) predict_regress(x DataFrame) (y DataFrame) {
// 	return
// }

// func (knn *Knn) predict_classify(x DataFrame) (y DataFrame) {
// 	return
// }

// func (knn *Knn) fit(x DataFrame, y DataFrame) (err error) {
//
// 	knn.in_shape, err = x.CalcShape()
// 	if err != nil {
// 		return err
// 	}
//
// 	knn.out_shape, err = y.CalcShape()
// 	if err != nil {
// 		return err
// 	}
//
// 	knn.train_x = x
// 	knn.train_y = y
//
// 	return err
// }
