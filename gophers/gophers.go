package gophers

import (
	"errors"
	"fmt"
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

func SerArrToDf(dta []Series) DataFrame {

	out := DataFrame{
		data: dta,
	}

	out.shape, out.err = out.CalcShape()

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
