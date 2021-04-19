(defun line-extended-matrix (a k v)
    (let ((m (array-dimension a 0))
        (n (array-dimension a 1)))
        (let ((b (make-array (list (+ 1 m) n)))) ; b = Matrix(* (1 + m) n)
            (dotimes (i k)
                (dotimes (j n)
                    (setf (aref b i j) (aref a i j)))) ; b[i][j] = a[i][j]
            (loop
                :for j :below n
                :do (setf (aref b k j) (aref v j))) ; b[k][j] = v[j], j : 0 to n
            (dotimes (j n)
                (loop :for i :from k :to (- m 1) :do
                    (setf (aref b (+ 1 i) j) (aref a i j)))) ; b[i + 1][j] = a[i][j]
        b)
    )
)

(defun extended-matrix (a u i)
    (array-dimension a 0)
    (array-dimension a 1)
    (if (and (<= i (array-dimension a 0)) (>= i 0)) (line-extended-matrix a i u))
)

;test_variables
(defvar m1 (make-array '(3 3) :initial-contents '((0 0 0) (0 0 0) (0 0 0))))
(defvar u1 #(1 1 1))
(defvar m2 (make-array '(2 4) :initial-contents '((0 0 0 0) (0 0 0 0))))
(defvar u2 #(1 1 1 1))
(defvar m3 (make-array '(3 4) :initial-contents '((0 0 0 0) (0 0 0 0) (0 0 0 0))))
(defvar u3 #(1 1 1 1))
(defvar m4 (make-array '(1 1) :initial-contents '((0))))
(defvar u4 #(1))
(defvar m5 (make-array '(4 2) :initial-contents '((0 0) (0 0) (0 0) (0 0))))
(defvar u5 #(1 1))

;(extended-matrix m1 u1 0)
;(extended-matrix m2 u2 1)
;(extended-matrix m3 u3 2)
;(extended-matrix m4 u4 0)
;(extended-matrix m4 u4 1)
;(extended-matrix m4 u4 2)
;(extended-matrix m5 u5 0)
;(extended-matrix m5 u5 4)