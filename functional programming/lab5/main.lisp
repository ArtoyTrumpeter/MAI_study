(defun square (x) (* x x))

(defclass cart ()                ; имя класса и надклассы
  ((x :initarg :x :reader cart-x)   ; дескриптор слота x
   (y :initarg :y :reader cart-y))) ; дескриптор слота y


(defmethod print-object ((c cart) stream)
    (format stream "[CART x:~d y:~d]"
        (cart-x c) (cart-y c)
    )
)

(defclass polar ()
  ((radius :initarg :radius :accessor radius) 	; длина >=0
   (angle  :initarg :angle  :accessor angle)))	; угол (-pi;pi]


(defmethod print-object ((p polar) stream)
  (format stream "[POLAR radius ~d angle ~d]"
          (radius p) (angle p)))


(defmethod radius ((c cart))
  (sqrt (+ (square (cart-x c))
           (square (cart-y c)))))


(defmethod angle ((c cart))
  (atan (cart-y c) (cart-x c)))	; atan2 в Си


(defmethod cart-x ((p polar))
  (* (radius p) (cos (angle p))))


(defmethod cart-y ((p polar))
  (* (radius p) (sin (angle p))))

(defun on-single-line3 (data)
	(<= (abs (- (* (- (cart-x (second data)) (cart-x (first data)))
	   	   (- (cart-y (third data)) (cart-y (first data))))
	       (* (- (cart-x (third data)) (cart-x (first data)))
	       (- (cart-y (second data)) (cart-y (first data)))))
	    ) (fourth data)
	)
)

(defgeneric on-single-line3-p (v1 v2 v3 &optional tolerance))

(defmethod on-single-line3-p ((v1 cart) (v2 cart) (v3 cart) &optional (tolerance 0.001))  
	(on-single-line3 (list v1 v2 v3 tolerance)))

(defmethod on-single-line3-p ((v1 polar) (v2 polar) (v3 polar) &optional (tolerance 0.001))  
	(on-single-line3 (list v1 v2 v3 tolerance)))