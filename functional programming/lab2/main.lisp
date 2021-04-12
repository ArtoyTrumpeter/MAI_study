(defun nth-element (i list)
    (if (< i (list-length list))
        (find-el i list)
    )
)

(defun find-el (index list)
    (cond
        ((zerop index) (car list))
        (t (find-el (- index 1) (cdr list)))
    )
)