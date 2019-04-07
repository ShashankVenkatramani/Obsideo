//
//  RegisterSignIngViewController.swift
//  Matador
//
//  Created by Shanky(Prgm) on 4/6/19.
//  Copyright © 2019 Shashank Venkatramani. All rights reserved.
//

import UIKit
import Firebase
import FirebaseAuth

class RegisterSignIngViewController: UIViewController {

    @IBOutlet weak var email: UITextField!
    
    @IBOutlet weak var password: UITextField!
    
    @IBAction func signUpPressed(_ sender: Any) {
        Auth.auth().createUser(withEmail: email.text!, password: password.text!) { (user, error) in
            
        }
    }
    
    @IBAction func SignInPressed(_ sender: Any) {
        Auth.auth().signIn(withEmail: email.text!, password: password.text!) { (user, error) in
            
        }
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
    }
}
