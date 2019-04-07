//
//  PostTransactionViewController.swift
//  Matador
//
//  Created by Shanky(Prgm) on 4/6/19.
//  Copyright © 2019 Shashank Venkatramani. All rights reserved.
//

import UIKit
import LocalAuthentication
import Firebase
import FirebaseDatabase
class PostTransactionViewController: UIViewController {
    var uid:String = ""
    @IBOutlet weak var targetField: UITextField!
    @IBOutlet weak var amountField: UITextField!
    @IBAction func sendPressed(_ sender: Any) {
        let context:LAContext = LAContext()
        
        if context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: nil){
            context.evaluatePolicy(LAPolicy.deviceOwnerAuthenticationWithBiometrics, localizedReason: "Authorize payment") { (pass, error) in
                if pass {
                    let transaction = self.uid + self.targetField.text! + self.amountField.text!
                    let database = Database.database().reference()
                    
                    database.child("request")
                } else {
                    print("Failed")
                }
            }
        } else {
            
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
}
