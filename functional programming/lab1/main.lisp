(defun triangle-p (a b c)
    (cond
        ((> c (+ a b)) Nil)
        ((> a (+ b c)) Nil)
        ((> b (+ a c)) Nil)
        ((atom 1) T)
    )
)